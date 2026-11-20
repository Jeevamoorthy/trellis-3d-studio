"""
Hunyuan3D-2 & Hunyuan3D-2mv Shape Generation Runner
Runs Hunyuan3DDiTFlowMatchingPipeline with CPU offload for 8GB VRAM (RTX 4060).
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image

def preprocess_image_1024(img_path_or_pil):
    """
    4. Preprocess: rembg background removal, center, pad to a square RGBA at 1024px.
    """
    from rembg import remove

    if isinstance(img_path_or_pil, str):
        raw_img = Image.open(img_path_or_pil)
    else:
        raw_img = img_path_or_pil

    # Background removal if not already transparent
    has_alpha = False
    if raw_img.mode == 'RGBA':
        alpha_arr = np.array(raw_img)[:, :, 3]
        if not np.all(alpha_arr == 255):
            has_alpha = True

    if not has_alpha:
        print("  [Preprocess] Removing background via rembg...")
        cleaned = remove(raw_img.convert('RGB'))
    else:
        cleaned = raw_img

    img_rgba = cleaned.convert('RGBA')
    np_img = np.array(img_rgba)
    alpha = np_img[:, :, 3]

    # Crop to bounding box of character
    coords = np.argwhere(alpha > 10)
    if len(coords) > 0:
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        cropped = img_rgba.crop((xmin, ymin, xmax + 1, ymax + 1))
    else:
        cropped = img_rgba

    # Center and pad to 1024x1024 square RGBA canvas
    w, h = cropped.size
    max_dim = max(w, h)
    scale = (1024.0 * 0.90) / max_dim
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    paste_x = (1024 - new_w) // 2
    paste_y = (1024 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), mask=resized)
    print(f"  [Preprocess] Preprocessed to square 1024x1024 RGBA (character: {new_w}x{new_h})")
    return canvas


def texture_mesh_from_image(mesh, image_path, y_offset=0.0, scale_factor=1.0):
    """Projects high-res reference image colors onto 3D mesh vertices for photo-realistic facial and clothing detail."""
    try:
        from scipy.ndimage import distance_transform_edt
        raw_img = Image.open(image_path)
        if raw_img.mode != 'RGBA':
            from rembg import remove
            raw_img = remove(raw_img)
        rgba_arr = np.array(raw_img)

        # 1. Detect foreground mask (alpha > 40 and not solid white)
        is_fg = (rgba_arr[:, :, 3] > 40) & ((rgba_arr[:, :, 0] < 245) | (rgba_arr[:, :, 1] < 245) | (rgba_arr[:, :, 2] < 245))
        y_idx, x_idx = np.where(is_fg)
        if len(y_idx) == 0:
            return mesh

        # Nearest-neighbor foreground map to eliminate white edge-bleed
        indices = distance_transform_edt(~is_fg, return_distances=False, return_indices=True)
        nearest_y, nearest_x = indices[0], indices[1]
        fg_filled_rgb = rgba_arr[:, :, :3].copy()
        fg_filled_rgb[~is_fg] = rgba_arr[nearest_y[~is_fg], nearest_x[~is_fg], :3]

        y_min, y_max = y_idx.min(), y_idx.max()
        x_min, x_max = x_idx.min(), x_idx.max()

        verts = mesh.vertices.copy()
        normals = mesh.vertex_normals

        head_top = verts[:, 1].max()
        feet_bottom = verts[:, 1].min()
        total_height = max(1e-5, head_top - feet_bottom)
        x_span_3d = max(1e-5, verts[:, 0].max() - verts[:, 0].min())
        x_center_3d = (verts[:, 0].min() + verts[:, 0].max()) / 2.0

        aspect_3d = total_height / x_span_3d
        aspect_2d = (y_max - y_min) / max(1e-5, x_max - x_min)
        x_center_img = (x_min + x_max) / 2.0

        if aspect_3d < 1.6 and aspect_2d > 1.8:
            head_3d_mask = verts[:, 1] > (head_top - total_height * 0.22)
            head_width_3d = max(1e-4, verts[head_3d_mask, 0].ptp()) if head_3d_mask.any() else x_span_3d * 0.4
            head_2d_mask = y_idx < (y_min + (y_max - y_min) * 0.20)
            head_width_2d = max(10, x_idx[head_2d_mask].ptp()) if head_2d_mask.any() else (x_max - x_min) * 0.4
            
            scale = (head_width_2d / head_width_3d) * scale_factor
            scale_y = scale
            scale_x = scale

            img_y = y_min + (head_top - verts[:, 1]) * scale_y + y_offset * (y_max - y_min)
            img_x = x_center_img + (verts[:, 0] - x_center_3d) * scale_x
        else:
            scale_y = ((y_max - y_min) / total_height) * scale_factor
            scale_x = scale_y
            img_y = y_max - (verts[:, 1] - feet_bottom) * scale_y + y_offset * (y_max - y_min)
            img_x = x_center_img + (verts[:, 0] - x_center_3d) * scale_x

        img_x_c = np.clip(np.round(img_x).astype(int), 0, rgba_arr.shape[1] - 1)
        img_y_c = np.clip(np.round(img_y).astype(int), 0, rgba_arr.shape[0] - 1)

        front_colors = fg_filled_rgb[img_y_c, img_x_c]

        # Sample regional palettes from image foreground
        head_sub = fg_filled_rgb[y_min:y_min + int((y_max-y_min)*0.2), x_min:x_max]
        hair_col = head_sub.mean(axis=(0, 1)).astype(int)

        torso_sub = fg_filled_rgb[y_min + int((y_max-y_min)*0.25):y_min + int((y_max-y_min)*0.60), x_min:x_max]
        cloth_col = torso_sub.mean(axis=(0, 1)).astype(int)

        lower_sub = fg_filled_rgb[y_min + int((y_max-y_min)*0.60):y_max, x_min:x_max]
        lower_col = lower_sub.mean(axis=(0, 1)).astype(int)

        back_colors = np.zeros((len(verts), 3), dtype=np.uint8)
        y_rel = (verts[:, 1] - feet_bottom) / total_height

        back_colors[y_rel > 0.72] = hair_col
        back_colors[(y_rel <= 0.72) & (y_rel > 0.40)] = cloth_col
        back_colors[y_rel <= 0.40] = lower_col

        nz = normals[:, 2]
        w = np.clip((nz - (-0.20)) / 0.40, 0.0, 1.0)
        w = (1.0 - np.cos(w * np.pi)) / 2.0
        w = w[:, np.newaxis]

        final_rgb = (front_colors.astype(float) * w + back_colors.astype(float) * (1.0 - w)).astype(np.uint8)
        final_rgba = np.zeros((len(verts), 4), dtype=np.uint8)
        final_rgba[:, :3] = final_rgb
        final_rgba[:, 3] = 255

        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=final_rgba)
        print("  [Texturing] Projected reference image colors onto 3D mesh vertices without edge bleed.")
    except Exception as e:
        print(f"  [Texturing Warning] Could not project colors: {e}")
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    return mesh


def run_hunyuan_generation(
    front_image_path,
    output_glb_path,
    side_image_path=None,
    back_image_path=None,
    model_repo="tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv",
    octree_resolution=384,
    steps=50,
    guidance_scale=5.5,
    seed=42,
    progress_callback=None
):
    import hy3dgen
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FloaterRemover,
        DegenerateFaceRemover,
        FaceReducer
    )
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    import trimesh
    from trimesh.smoothing import filter_taubin

    print("=" * 65)
    print("HUNYUAN3D-2 / 2MV HIGH-FIDELITY SHAPE GENERATION")
    print(f"1. Pipeline Class: Hunyuan3DDiTFlowMatchingPipeline")
    print(f"   Checkpoint: {model_repo} (subfolder='{subfolder}')")
    print(f"   Hardware: RTX 4060 8GB VRAM (Accelerate Model CPU Offload)")
    print(f"   Inference Params: steps={steps}, guidance_scale={guidance_scale}, octree_resolution={octree_resolution}")
    print("=" * 65)

    if progress_callback: progress_callback(10, "Loading Hunyuan3D-2mv Pipeline & Weights...")

    t0 = time.time()
    # 2. Implement Hunyuan3D-2 shape generation via hy3dgen with CPU offload for 8GB VRAM
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_repo,
        subfolder=subfolder,
        use_safetensors=True,
        device='cuda'
    )
    print("[CPU Offload] Enabling accelerate model CPU offload for 8GB VRAM...")
    pipeline.enable_model_cpu_offload()

    # 4. Preprocess: rembg background removal, center, pad to a square RGBA at 1024px
    if progress_callback: progress_callback(25, "Preprocessing views (rembg + 1024px square)...")
    print("\n[Input Conditioning] Preprocessing input views:")
    image_dict = {}

    print(f"  Front view: {front_image_path}")
    image_dict['front'] = preprocess_image_1024(front_image_path)

    # 3. Add Hunyuan3D-2mv so the Front/Side/Back slots are real multi-view inputs
    if side_image_path and os.path.exists(side_image_path):
        print(f"  Side/Left view: {side_image_path}")
        image_dict['left'] = preprocess_image_1024(side_image_path)

    if back_image_path and os.path.exists(back_image_path):
        print(f"  Back view: {back_image_path}")
        image_dict['back'] = preprocess_image_1024(back_image_path)

    print(f"\nConditioning Views active in Hunyuan3D-2mv: {list(image_dict.keys())}")

    # Seed
    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))

    # Run Shape Generation
    if progress_callback: progress_callback(40, f"Running DiT Flow Matching ({steps} steps, octree={octree_resolution})...")
    print(f"\n[Generation] Starting DiT Flow Matching sampling ({steps} steps, guidance={guidance_scale})...")
    gen_t0 = time.time()
    outputs = pipeline(
        image=image_dict,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=8000,
        output_type='mesh'
    )
    print(f"[Generation] Diffusion complete in {time.time() - gen_t0:.1f}s")

    # Export raw mesh to trimesh
    raw_mesh = export_to_trimesh(outputs)[0]
    print(f"[Raw Output] Faces: {len(raw_mesh.faces):,}, Vertices: {len(raw_mesh.vertices):,}")

    # 5. Post-process with FloaterRemover, DegenerateFaceRemover and FaceReducer, then light smoothing only
    if progress_callback: progress_callback(85, "Post-processing: FloaterRemover, DegenerateRemover, FaceReducer...")
    print("\n[Post-Processing] Applying Tencent Hunyuan3D official post-processors:")
    floater_remover = FloaterRemover()
    degenerate_remover = DegenerateFaceRemover()
    face_reducer = FaceReducer()

    print("  Step 1: FloaterRemover (eliminating floating mesh shards)...")
    cleaned_mesh = floater_remover(raw_mesh)

    print("  Step 2: DegenerateFaceRemover (fixing zero-area/collapsed faces)...")
    cleaned_mesh = degenerate_remover(cleaned_mesh)

    print("  Step 3: FaceReducer (optimizing topology cleanly)...")
    cleaned_mesh = face_reducer(cleaned_mesh, max_facenum=350000)

    # Micro-smoothing (1 iteration gentle volume preservation to keep sharp eyelids, eyes, cloth seams and hair tips)
    print("  Step 4: Sharp feature preservation (1 gentle micro-pass)...")
    smoothed_mesh = filter_taubin(cleaned_mesh, lamb=0.12, nu=-0.13, iterations=1)
    smoothed_mesh.fix_normals()

    # Ensure single solid connected component (discard stray micro-islands)
    components = smoothed_mesh.split(only_watertight=False)
    if len(components) > 1:
        components = sorted(components, key=lambda c: len(c.faces), reverse=True)
        # Keep primary solid body
        smoothed_mesh = components[0]

    # Project reference image colors onto mesh vertices (eyes, face, hair, and clothing textures)
    print("  Step 5: Reference Image Color & Facial Texture Projection...")
    smoothed_mesh = texture_mesh_from_image(smoothed_mesh, front_image_path)

    # Verification stats
    final_components = smoothed_mesh.split(only_watertight=False)
    num_comps = len(final_components)
    num_faces = len(smoothed_mesh.faces)
    num_verts = len(smoothed_mesh.vertices)
    is_watertight = smoothed_mesh.is_watertight

    print("\n" + "=" * 65)
    print("VERIFICATION OF GENERATED 3D MODEL:")
    print(f"  • Total Faces: {num_faces:,}")
    print(f"  • Total Vertices: {num_verts:,}")
    print(f"  • Connected Components: {num_comps}")
    print(f"  • Watertight: {is_watertight}")
    print(f"  • Export Destination: {output_glb_path}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(output_glb_path)), exist_ok=True)
    smoothed_mesh.export(output_glb_path)
    print(f"[SUCCESS] Exported clean studio sculpt to: {output_glb_path}")

    # Also save verification stats to JSON for the desktop app HUD and report
    stats = {
        "pipeline_class": "Hunyuan3DDiTFlowMatchingPipeline",
        "checkpoint": f"{model_repo}/{subfolder}",
        "faces": num_faces,
        "vertices": num_verts,
        "connected_components": num_comps,
        "watertight": is_watertight,
        "total_time_seconds": round(time.time() - t0, 1)
    }
    with open(output_glb_path + ".stats.json", "w") as f:
        import json
        json.dump(stats, f, indent=2)

    if progress_callback: progress_callback(100, f"Ready! {num_faces:,} Faces, {num_comps} Component(s)")
    return smoothed_mesh, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D-2 Shape Generation")
    parser.add_argument("front_image", type=str, help="Front reference image path")
    parser.add_argument("output_glb", type=str, help="Output .glb file path")
    parser.add_argument("--side_image", type=str, default=None, help="Side reference image path")
    parser.add_argument("--back_image", type=str, default=None, help="Back reference image path")
    parser.add_argument("--octree_resolution", type=int, default=256, help="Octree resolution (256-384)")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    args = parser.parse_args()

    run_hunyuan_generation(
        front_image_path=args.front_image,
        output_glb_path=args.output_glb,
        side_image_path=args.side_image,
        back_image_path=args.back_image,
        octree_resolution=args.octree_resolution,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
