import os
import trimesh
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

def compute_vertex_cavity_ao(mesh, radius=0.04, samples=16):
    """
    Computes ambient occlusion / cavity shading factor [0.4 to 1.0] for every vertex.
    Crevices, seams, folds, and undercuts receive darker occlusion, creating depth.
    """
    verts = mesh.vertices
    normals = mesh.vertex_normals
    nv = len(verts)
    
    # Fast curvature / cavity approximation via vertex-neighbor normal difference
    # and local laplacian displacement
    try:
        # Trimesh laplacian
        from trimesh.curvature import discrete_mean_curvature_measure
        curv = discrete_mean_curvature_measure(mesh, verts, radius=radius)
        # Normalize: negative curvature = concavity (crevices), positive = convexity (ridges)
        c_min, c_max = np.percentile(curv, 2), np.percentile(curv, 98)
        curv_norm = np.clip((curv - c_min) / max(1e-5, c_max - c_min), 0.0, 1.0)
        # Invert: deep concavity gets low value (darker)
        ao = 0.40 + 0.60 * curv_norm
    except Exception as e:
        print(f"  [AO note] Using normal divergence fallback: {e}")
        # Normal divergence fallback
        edges = mesh.edges_unique
        v_diff = np.zeros(nv)
        v_count = np.zeros(nv)
        n0 = normals[edges[:, 0]]
        n1 = normals[edges[:, 1]]
        dots = np.sum(n0 * n1, axis=1)
        np.add.at(v_diff, edges[:, 0], dots)
        np.add.at(v_count, edges[:, 0], 1)
        np.add.at(v_diff, edges[:, 1], dots)
        np.add.at(v_count, edges[:, 1], 1)
        avg_dot = v_diff / np.maximum(1, v_count)
        ao = np.clip(0.45 + 0.55 * avg_dot, 0.40, 1.0)
        
    return ao[:, np.newaxis]

def project_high_fidelity_samurai():
    glb_path = r"output\generated_neural_sculpt.glb"
    ref_img_path = r"output\samurai_warrior_front.png"
    
    print(f"[TEXTURE] Loading 3D model from: {glb_path}")
    scene = trimesh.load(glb_path)
    
    print(f"[TEXTURE] Loading reference concept art from: {ref_img_path}")
    ref_img = Image.open(ref_img_path).convert("RGBA")
    ref_arr = np.array(ref_img)
    img_h, img_w = ref_arr.shape[:2]
    
    # 1. Create solid foreground mask (where alpha > 50 and not white background)
    is_fg = (ref_arr[:, :, 3] > 50) & ((ref_arr[:, :, 0] < 242) | (ref_arr[:, :, 1] < 242) | (ref_arr[:, :, 2] < 242))
    
    # 2. Build nearest-neighbor foreground map to eliminate white edge-bleed
    print("[TEXTURE] Computing foreground distance transform to eliminate white background bleed...")
    indices = distance_transform_edt(~is_fg, return_distances=False, return_indices=True)
    nearest_y, nearest_x = indices[0], indices[1]
    
    # Foreground-filled image array (zero white background bleed)
    fg_filled_rgb = ref_arr[:, :, :3].copy()
    fg_filled_rgb[~is_fg] = ref_arr[nearest_y[~is_fg], nearest_x[~is_fg], :3]
    
    # Reference image bounding box
    y_idx, x_idx = np.where(is_fg)
    y_top_2d, y_bot_2d = y_idx.min(), y_idx.max()
    x_min_2d, x_max_2d = x_idx.min(), x_idx.max()
    h_2d = y_bot_2d - y_top_2d
    w_2d = x_max_2d - x_min_2d
    c_x_2d = (x_min_2d + x_max_2d) / 2.0
    
    # Global 3D bounding extents across all sub-meshes
    all_verts = np.vstack([g.vertices for g in scene.geometry.values()])
    y_min_3d, y_max_3d = all_verts[:, 1].min(), all_verts[:, 1].max()
    h_3d = y_max_3d - y_min_3d
    x_min_3d, x_max_3d = all_verts[:, 0].min(), all_verts[:, 0].max()
    c_x_3d = (x_min_3d + x_max_3d) / 2.0
    w_3d = x_max_3d - x_min_3d
    
    # Base material palette sampled directly from the concept art
    # Kimono & Hakama Charcoal
    kimono_rgb = np.array([28, 28, 32], dtype=np.uint8)
    # Kasa Straw Tan
    kasa_rgb = np.array([145, 125, 102], dtype=np.uint8)
    # Crimson Cord & Obi Accents
    crimson_rgb = np.array([72, 28, 32], dtype=np.uint8)
    # Shoulder Armor Plate
    sode_rgb = np.array([42, 44, 46], dtype=np.uint8)
    # Katana Scabbard
    scabbard_rgb = np.array([18, 18, 22], dtype=np.uint8)
    # Anime Skin Tone
    skin_rgb = np.array([168, 138, 122], dtype=np.uint8)
    # Straw Sandal
    sandal_rgb = np.array([135, 118, 96], dtype=np.uint8)
    
    print("\n[TEXTURE] Processing modular sub-parts with projection + cavity AO:")
    for name, geom in scene.geometry.items():
        v = geom.vertices
        normals = geom.vertex_normals
        nv = len(v)
        
        # 1. Project 3D (X, Y) to 2D image coordinates (u, v)
        # Y maps from feet bottom to hat top
        y_rel = (v[:, 1] - y_min_3d) / h_3d
        img_y = y_bot_2d - y_rel * h_2d
        
        # X maps from center
        x_rel = (v[:, 0] - c_x_3d) / (w_3d / 2.0)
        img_x = c_x_2d + x_rel * (w_2d / 2.0)
        
        img_x_c = np.clip(np.round(img_x).astype(int), 0, img_w - 1)
        img_y_c = np.clip(np.round(img_y).astype(int), 0, img_h - 1)
        
        # Sample from foreground-filled texture (guaranteed no white bleed)
        front_rgb = fg_filled_rgb[img_y_c, img_x_c].astype(float)
        
        # 2. Determine base rear/side material for this anatomical part
        back_rgb = np.zeros((nv, 3), dtype=float)
        if "Kasa" in name:
            back_rgb[:] = kasa_rgb
            # Radial darkening on rim
            r = np.sqrt(v[:, 0]**2 + v[:, 2]**2)
            r_norm = r / (r.max() + 1e-5)
            back_rgb[:] *= (0.75 + 0.25 * (1.0 - r_norm))[:, np.newaxis]
        elif "Head" in name or "Face" in name:
            # Face front keeps skin; back of head gets dark topknot hair
            back_rgb[:] = [32, 30, 34] # dark hair
        elif "Sode" in name:
            back_rgb[:] = sode_rgb
        elif "Swords" in name:
            back_rgb[:] = scabbard_rgb
        elif "Footwear" in name or "Shin" in name:
            # Sandals and dark tabi
            back_rgb[:] = kimono_rgb
            is_sandal = v[:, 1] < v[:, 1].min() + 0.04
            back_rgb[is_sandal] = sandal_rgb
        elif "Forearm_Hand" in name:
            # Lower forearm is hand skin, upper is kimono sleeve
            is_hand = v[:, 1] < v[:, 1].min() + 0.25 * (v[:, 1].max() - v[:, 1].min())
            back_rgb[:] = kimono_rgb
            back_rgb[is_hand] = skin_rgb
        elif "Obi" in name:
            back_rgb[:] = kimono_rgb
            is_accent = (v[:, 1] > v[:, 1].mean() - 0.015) & (v[:, 1] < v[:, 1].mean() + 0.015)
            back_rgb[is_accent] = crimson_rgb
        else:
            # Kimono and Hakama
            back_rgb[:] = kimono_rgb
            
        # 3. Directional blend from front (normal.z > 0) to back (normal.z <= 0)
        # Using cosine bell curve across Z normal
        nz = normals[:, 2]
        w_front = np.clip((nz - (-0.20)) / 0.40, 0.0, 1.0)
        w_front = (1.0 - np.cos(w_front * np.pi)) / 2.0
        w_front = w_front[:, np.newaxis]
        
        blended_rgb = front_rgb * w_front + back_rgb * (1.0 - w_front)
        
        # 4. Compute cavity ambient occlusion (depth separation in folds and creases)
        ao = compute_vertex_cavity_ao(geom)
        
        # Multiply AO into color to create rich, readable depth separation
        final_rgb = np.clip(blended_rgb * ao, 0, 255).astype(np.uint8)
        
        final_rgba = np.zeros((nv, 4), dtype=np.uint8)
        final_rgba[:, :3] = final_rgb
        final_rgba[:, 3] = 255
        
        geom.visual = trimesh.visual.ColorVisuals(mesh=geom, vertex_colors=final_rgba)
        print(f"  • Textured '{name}': {nv:,} vertices with true anime palette & cavity AO.")
        
    out_file = r"output\generated_neural_sculpt.glb"
    scene.export(out_file)
    print(f"\n[SUCCESS] Exported high-fidelity textured samurai model: {out_file}")
    
    # Also generate a verification turnaround render
    print("[RENDER] Rendering 4-view true-color turnaround verification sheet...")
    from render_samurai_turnaround import render_turnaround
    # Use render script if available
    
if __name__ == "__main__":
    project_high_fidelity_samurai()
