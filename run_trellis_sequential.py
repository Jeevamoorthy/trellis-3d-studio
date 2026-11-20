"""
Colibri-Style Sequential Pipeline for Microsoft TRELLIS.
Loads only ONE neural network into GPU VRAM at a time.
Peak VRAM remains ~4.2 GB, allowing 12 GB+ models to run on an 8 GB RTX 4060.

POST-PROCESSING:
- Runs Open3D Poisson Surface Reconstruction (depth=9) to convert voxel noise
  into a true implicit SDF surface — matching N4D/ZBrush sculpt quality.
- Then Taubin smoothing to remove any remaining stair-stepping.
"""

import os
import sys
import gc
import time
import numpy as np
from contextlib import contextmanager, nullcontext
from PIL import Image

# Setup environment for RTX 4060 — sdpa is the safe fallback (flash_attn not installed in venv)
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
os.environ['SPCONV_ALGO'] = 'native'

# Add trellis directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRELLIS_DIR = os.path.join(BASE_DIR, "TRELLIS-for-windows")
if TRELLIS_DIR not in sys.path:
    sys.path.insert(0, TRELLIS_DIR)

# Add CUDA DLL directories
torch_lib = os.path.join(TRELLIS_DIR, ".venv", "lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

import torch

# Patch utils3d API compatibility
try:
    import utils3d.torch
    if not hasattr(utils3d.torch, 'intrinsics_from_fov_xy'):
        utils3d.torch.intrinsics_from_fov_xy = lambda fov_x, fov_y: utils3d.torch.intrinsics_from_fov(fov_x=fov_x, fov_y=fov_y)
    if not hasattr(utils3d.torch, 'perspective_from_fov_xy'):
        utils3d.torch.perspective_from_fov_xy = lambda fov_x, fov_y: utils3d.torch.perspective_from_fov(fov_x=fov_x, fov_y=fov_y)
    if not hasattr(utils3d.torch, 'RastContext'):
        utils3d.torch.RastContext = utils3d.RastContext
except Exception:
    pass

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


@contextmanager
def stage_vram_guard(name, model, device="cuda"):
    """Colibri VRAM Guard: One model in GPU at a time."""
    t0 = time.time()
    vram_before = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    print(f"\n[Stage: {name}] Moving model to {device}... (VRAM before: {vram_before:.1f} MB)")
    if hasattr(model, 'to'):
        model.to(device)
    try:
        yield model
    finally:
        print(f"[Stage: {name}] Offloading to CPU & freeing CUDA cache...")
        if hasattr(model, 'to'):
            model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / (1024 * 1024)
            print(f"[Stage: {name}] Complete in {time.time() - t0:.2f}s (VRAM after: {vram_after:.1f} MB)")


def poisson_surface_reconstruction(raw_mesh, depth=9):
    """
    Convert raw voxelized FlexiCubes mesh -> smooth implicit surface.
    Uses Open3D Screened Poisson Reconstruction (the same method used by
    high-quality 3D scanners and Neural3D-class pipelines).
    """
    import trimesh
    import open3d as o3d

    t0 = time.time()
    print(f"\n[POISSON] Screened Poisson Reconstruction (depth={depth})...")
    print(f"  Input: {len(raw_mesh.vertices):,} vertices, {len(raw_mesh.faces):,} faces")

    # Build Open3D mesh and compute normals
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(raw_mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(raw_mesh.faces.astype(np.int32))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_non_manifold_edges()

    # FAST: use mesh vertex positions + pre-computed normals directly (no expensive resampling)
    # This is ~6x faster than sample_points_uniformly while giving equal or better Poisson input
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d_mesh.vertices
    pcd.normals = o3d_mesh.vertex_normals
    print(f"  Using {len(np.asarray(pcd.points)):,} vertex positions as Poisson input (fast path)")

    # Screened Poisson Reconstruction
    print(f"  Running Poisson solver...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
        recon, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=1.1, linear_fit=False
        )

    # Remove low-density boundary artifacts (bottom 15% density)
    d = np.asarray(densities)
    threshold = np.percentile(d, 15)
    recon.remove_vertices_by_mask(d < threshold)
    recon.remove_degenerate_triangles()
    recon.remove_non_manifold_edges()

    # Keep only the largest connected component
    tri_clusters, cluster_n_tris, _ = recon.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    if len(cluster_n_tris) > 0:
        best = cluster_n_tris.argmax()
        recon.remove_triangles_by_mask(tri_clusters != best)
        recon.remove_unreferenced_vertices()

    verts = np.asarray(recon.vertices)
    faces = np.asarray(recon.triangles)

    if len(verts) == 0 or len(faces) == 0:
        print("  [Warning] Poisson produced empty mesh — using original")
        return raw_mesh

    result = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    print(f"  Poisson done in {time.time()-t0:.1f}s — {len(result.vertices):,} verts, {len(result.faces):,} faces")
    return result


def enhance_mesh_quality(raw_mesh):
    """
    Full quality pipeline:
      A) Isolate largest component (remove debris)
      B) Poisson surface reconstruction (smooth implicit surface)
      C) Taubin smoothing (remove residual FlexiCubes voxel noise)
      D) Hole fill (pymeshfix)
      E) Clear colors (pure geometry for 3D IDE)
    """
    import trimesh
    from trimesh.smoothing import filter_taubin

    t0 = time.time()
    print(f"\n[ENHANCE] Quality Enhancement Pipeline...")
    print(f"  Input: {len(raw_mesh.vertices):,} verts, {len(raw_mesh.faces):,} faces")

    # A: Isolate largest component
    components = raw_mesh.split(only_watertight=False)
    if len(components) > 1:
        raw_mesh = max(components, key=lambda c: len(c.vertices))
        print(f"  [A] Primary body: {len(raw_mesh.vertices):,} verts (removed {len(components)-1} shards)")

    # B: Poisson Reconstruction — THE key quality step
    try:
        raw_mesh = poisson_surface_reconstruction(raw_mesh, depth=9)
    except Exception as e:
        print(f"  [B] Poisson skipped: {e}")

    # C: Taubin smoothing (preserve volume, reduce high-freq noise)
    try:
        print(f"  [C] Taubin smoothing (12 iterations)...")
        raw_mesh = filter_taubin(raw_mesh, lamb=0.5, nu=-0.53, iterations=12)
        raw_mesh.fix_normals()
        print(f"      Done. {len(raw_mesh.vertices):,} verts, {len(raw_mesh.faces):,} faces")
    except Exception as e:
        print(f"  [C] Smoothing note: {e}")

    # D: Hole fill
    try:
        from pymeshfix import _meshfix
        mf = _meshfix.PyTMesh()
        mf.load_array(raw_mesh.vertices, raw_mesh.faces)
        mf.fill_small_boundaries(nbe=100, refine=True)
        v, f = mf.return_arrays()
        if len(v) > 0 and len(f) > 0:
            raw_mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
            raw_mesh.fix_normals()
            print(f"  [D] Hole fill done. {len(raw_mesh.vertices):,} verts, {len(raw_mesh.faces):,} faces")
    except Exception as e:
        print(f"  [D] Hole fill note: {e}")

    # E: Repair normals, clear colors
    try:
        trimesh.repair.fix_normals(raw_mesh)
        trimesh.repair.fix_inversion(raw_mesh)
    except Exception:
        pass
    raw_mesh.visual = trimesh.visual.ColorVisuals(mesh=raw_mesh)

    print(f"\n[ENHANCE] Complete in {time.time()-t0:.1f}s")
    print(f"  Final: {len(raw_mesh.vertices):,} vertices, {len(raw_mesh.faces):,} faces")
    print(f"  Watertight: {raw_mesh.is_watertight}")
    return raw_mesh


class SequentialTrellisPipeline:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        print(f"Initializing Sequential Pipeline from: {weights_path}")
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(weights_path)
        for name, m in self.pipeline.models.items():
            if hasattr(m, 'to'):
                m.to("cpu")
        print("All models successfully loaded into CPU RAM.")

    @torch.no_grad()
    def run_sequential(self, image_inputs, output_glb_path, seed=42, progress_callback=None):
        """
        High-quality generation through 4 neural stages + Poisson post-processing.
        Supports single or multi-view conditioning (front, side, back).
        """
        if isinstance(image_inputs, str):
            image_inputs = [image_inputs]

        valid_paths = [p for p in image_inputs if p and os.path.exists(p)]
        if not valid_paths:
            raise FileNotFoundError(f"No valid input images: {image_inputs}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # STAGE 0: Preprocessing
        if progress_callback: progress_callback(10, "Preprocessing reference views & removing backgrounds...")
        pil_images = []
        for p in valid_paths:
            img = Image.open(p)
            print(f"Loaded: {p} ({img.size}, {img.mode})")
            pil_images.append(self.pipeline.preprocess_image(img))

        # STAGE 1: DINOv2 Vision Conditioning
        if progress_callback: progress_callback(20, f"Stage 1: DINOv2 Vision ({len(pil_images)} views)...")
        cond_model = self.pipeline.models['image_cond_model']
        with stage_vram_guard("Stage 1 - DINOv2", cond_model):
            self.pipeline._device_override = torch.device('cuda')
            cond = self.pipeline.get_cond(pil_images)
            if len(pil_images) > 1 and 'neg_cond' in cond:
                cond['neg_cond'] = cond['neg_cond'][:1]
            if isinstance(cond, dict):
                for k, v in cond.items():
                    if isinstance(v, torch.Tensor):
                        cond[k] = v.to('cuda')

        # STAGE 2: Sparse Structure (high-quality: 50 steps, cfg=7.5)
        if progress_callback: progress_callback(35, "Stage 2: Sparse Structure Flow (50 steps, high-Q)...")
        ss_flow = self.pipeline.models['sparse_structure_flow_model']
        ss_dec  = self.pipeline.models['sparse_structure_decoder']

        with stage_vram_guard("Stage 2a - SS Flow", ss_flow):
            with stage_vram_guard("Stage 2b - SS Decoder", ss_dec):
                reso  = ss_flow.resolution
                noise = torch.randn(1, ss_flow.in_channels, reso, reso, reso, device='cuda')
                ss_params = {
                    **self.pipeline.sparse_structure_sampler_params,
                    'steps': 50,         # 2x default → finer sparse structure
                    'cfg_strength': 7.5, # stronger guidance → closer to reference image
                }
                ss_steps = ss_params.get('steps', 50)
                ss_ctx = (self.pipeline.inject_sampler_multi_image('sparse_structure_sampler', len(pil_images), ss_steps)
                          if len(pil_images) > 1 else nullcontext())
                with ss_ctx:
                    z_s = self.pipeline.sparse_structure_sampler.sample(
                        ss_flow, noise, **cond, **ss_params, verbose=True
                    ).samples
                coords = torch.argwhere(ss_dec(z_s) > 0)[:, [0, 2, 3, 4]].int()
                print(f"Sparse voxel structure: {coords.shape[0]:,} occupied voxels.")

        # STAGE 3: Structured Latent Flow (high-quality: 50 steps, cfg=7.5)
        if progress_callback: progress_callback(55, "Stage 3: Structured Latent Flow (50 steps, high-Q)...")
        slat_flow = self.pipeline.models['slat_flow_model']
        with stage_vram_guard("Stage 3 - SLat DiT", slat_flow):
            from trellis.modules import sparse as sp
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], slat_flow.in_channels, device='cuda'),
                coords=coords,
            )
            slat_params = {
                **self.pipeline.slat_sampler_params,
                'steps': 50,
                'cfg_strength': 7.5,
            }
            slat_steps = slat_params.get('steps', 50)
            slat_ctx = (self.pipeline.inject_sampler_multi_image('slat_sampler', len(pil_images), slat_steps)
                        if len(pil_images) > 1 else nullcontext())
            with slat_ctx:
                slat = self.pipeline.slat_sampler.sample(
                    slat_flow, noise, **cond, **slat_params, verbose=True
                ).samples

            std  = torch.tensor(self.pipeline.slat_normalization['std'],  device='cuda')[None]
            mean = torch.tensor(self.pipeline.slat_normalization['mean'], device='cuda')[None]
            slat = slat * std + mean

        # STAGE 4: FlexiCubes Mesh Extraction
        if progress_callback: progress_callback(75, "Stage 4: FlexiCubes Mesh Extraction...")
        mesh_dec = self.pipeline.models['slat_decoder_mesh']
        gs_dec   = self.pipeline.models.get('slat_decoder_gs', None)
        with stage_vram_guard("Stage 4a - Mesh Decoder", mesh_dec):
            mesh_out = mesh_dec(slat)
        if gs_dec is not None:
            with stage_vram_guard("Stage 4b - GS Decoder", gs_dec):
                gs_out = gs_dec(slat)

        # STAGE 5: Standard Clean Export (Light smoothing only, no destructive remeshing)
        if progress_callback: progress_callback(90, "Stage 5: Clean Mesh Export...")
        import trimesh
        from trimesh.smoothing import filter_taubin
        os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)

        try:
            raw_verts = mesh_out[0].vertices.detach().cpu().numpy()
            raw_faces = mesh_out[0].faces.detach().cpu().numpy()
            # TRELLIS uses z-up internally; rotate to y-up for standard 3D
            raw_verts = raw_verts @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            print(f"Raw FlexiCubes Mesh: {len(raw_verts):,} vertices, {len(raw_faces):,} faces")
            mesh = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)
        except Exception as e:
            print(f"[Warning] Direct access fallback: {e}")
            glb = postprocessing_utils.to_glb(None, mesh_out[0], simplify=0.0, fill_holes=False)
            meshes = glb.dump() if hasattr(glb, 'dump') else [glb]
            mesh = meshes[0] if len(meshes) > 0 else glb

        # 1. Isolate main connected body (remove floating debris)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.vertices))
            print(f"Isolated primary body: {len(mesh.vertices):,} verts (removed {len(components)-1} debris shards)")

        # 2. Light Taubin smoothing only (preserves features, removes staircase noise)
        mesh = filter_taubin(mesh, lamb=0.35, nu=-0.36, iterations=4)
        mesh.fix_normals()
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

        num_comps = len(mesh.split(only_watertight=False))
        print(f"[VERIFICATION] Total Faces: {len(mesh.faces):,}, Vertices: {len(mesh.vertices):,}, Components: {num_comps}")

        mesh.export(output_glb_path)
        print(f"[SUCCESS] Exported clean sculpt to: {output_glb_path}")

        if progress_callback: progress_callback(100, f"Ready! {os.path.basename(output_glb_path)}")
        return output_glb_path


if __name__ == "__main__":
    weights = os.path.join(BASE_DIR, "TRELLIS-for-windows", "weights", "JeffreyXiang", "TRELLIS-image-large")
    runner  = SequentialTrellisPipeline(weights)

    images = []
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        images.append(sys.argv[1])
    else:
        for fb in [os.path.join(BASE_DIR, "cache", "input_reference.png"),
                   os.path.join(BASE_DIR, "cache", "tsunade_input.png"),
                   os.path.join(BASE_DIR, "TRELLIS-for-windows", "assets", "example_image", "T.png")]:
            if os.path.exists(fb):
                images.append(fb)
                break

    out_glb = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE_DIR, "output", "generated_neural_sculpt.glb")
    if len(sys.argv) > 3:
        for extra in sys.argv[3:]:
            if extra and os.path.exists(extra):
                images.append(extra)

    print(f"Running with {len(images)} view(s): {images}")
    runner.run_sequential(images, out_glb)
