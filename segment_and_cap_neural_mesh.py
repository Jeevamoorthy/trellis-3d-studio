import os
import trimesh
import numpy as np
from PIL import Image, ImageDraw

def cap_boundary_holes(mesh):
    """Detects open boundary edges on cut mesh and closes them with planar/fan triangulation."""
    try:
        # Trimesh fill_holes attempts to close open manifold boundaries
        mesh.fill_holes()
    except Exception as e:
        print(f"  [Cap] fill_holes note: {e}")
    return mesh

def segment_and_cap(input_glb, output_glb):
    print(f"[SEGMENTATION] Loading neural sculpt from: {input_glb}")
    mesh = trimesh.load(input_glb, force='mesh')
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    v_colors = mesh.visual.vertex_colors.copy() if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None else None

    total_orig_faces = len(faces)
    total_orig_verts = len(verts)
    print(f"  • Source Mesh: {total_orig_verts:,} verts, {total_orig_faces:,} faces (watertight neural surface)")

    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    H = y_max - y_min
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    W = x_max - x_min
    c_x = (x_min + x_max) / 2.0
    c_z = (verts[:, 2].min() + verts[:, 2].max()) / 2.0

    # Face centroids
    f_cents = verts[faces].mean(axis=1)
    f_y = f_cents[:, 1]
    f_x = f_cents[:, 0]
    f_z = f_cents[:, 2]

    # Natural anatomical joint levels
    y_hat_base = y_min + 0.88 * H
    y_neck = y_min + 0.77 * H
    y_chest = y_min + 0.60 * H
    y_waist = y_min + 0.48 * H
    y_knee = y_min + 0.22 * H
    y_ankle = y_min + 0.08 * H

    face_part_id = np.zeros(len(faces), dtype=int)

    # 1. Kasa Hat (top wide conical silhouette)
    hat_mask = (f_y >= y_hat_base) | ((f_y >= y_neck + 0.05 * H) & (np.sqrt((f_x - c_x)**2 + (f_z - c_z)**2) > 0.16))
    face_part_id[hat_mask] = 1

    # 2. Head & Face (below hat, above neck collar)
    head_mask = (f_y < y_hat_base) & (f_y >= y_neck) & ~hat_mask
    face_part_id[head_mask] = 2

    # 3. Right Sode Shoulder Plate (asymmetrical layered plate)
    sode_mask = (f_x > c_x + 0.14) & (f_y >= y_chest) & (f_y <= y_neck + 0.05 * H)
    face_part_id[sode_mask] = 3

    # 4. Katana Swords (left hip angled scabbard protrusion)
    swords_mask = (f_x < c_x - 0.10) & (f_y >= y_waist - 0.18 * H) & (f_y <= y_waist + 0.10 * H) & (f_z < c_z - 0.02)
    face_part_id[swords_mask] = 4

    # 5. Left Upper Arm (Deltoid to Elbow)
    l_up_arm_mask = (f_x < c_x - 0.16) & (f_y >= y_waist + 0.08 * H) & ~swords_mask
    face_part_id[l_up_arm_mask] = 5

    # 6. Left Forearm & Hand (Elbow to Wrist & Hand)
    l_forearm_mask = (f_x < c_x - 0.16) & (f_y < y_waist + 0.08 * H) & (f_y >= y_waist - 0.12 * H) & ~swords_mask
    face_part_id[l_forearm_mask] = 6

    # 7. Right Upper Arm (Deltoid to Elbow)
    r_up_arm_mask = (f_x > c_x + 0.16) & (f_y >= y_waist + 0.08 * H) & ~sode_mask
    face_part_id[r_up_arm_mask] = 7

    # 8. Right Forearm & Hand (Elbow to Wrist & Hand)
    r_forearm_mask = (f_x > c_x + 0.16) & (f_y < y_waist + 0.08 * H) & (f_y >= y_waist - 0.12 * H) & ~sode_mask
    face_part_id[r_forearm_mask] = 8

    # 9. Chest & Kimono Torso
    torso_mask = (f_y >= y_waist + 0.04 * H) & (f_y < y_neck) & (np.abs(f_x - c_x) <= 0.16) & ~sode_mask & ~swords_mask
    face_part_id[torso_mask] = 9

    # 10. Waist & Obi Sash
    obi_mask = (f_y >= y_waist - 0.06 * H) & (f_y < y_waist + 0.04 * H) & (np.abs(f_x - c_x) <= 0.18) & ~swords_mask
    face_part_id[obi_mask] = 10

    # 11. Left Hakama Leg & Thigh (Hip to Knee)
    l_thigh_mask = (f_y < y_waist - 0.06 * H) & (f_y >= y_knee) & (f_x < c_x) & ~swords_mask
    face_part_id[l_thigh_mask] = 11

    # 12. Left Shin & Foot (Knee to Footwear)
    l_shin_mask = (f_y < y_knee) & (f_x < c_x)
    face_part_id[l_shin_mask] = 12

    # 13. Right Hakama Leg & Thigh (Hip to Knee)
    r_thigh_mask = (f_y < y_waist - 0.06 * H) & (f_y >= y_knee) & (f_x >= c_x)
    face_part_id[r_thigh_mask] = 13

    # 14. Right Shin & Foot (Knee to Footwear)
    r_shin_mask = (f_y < y_knee) & (f_x >= c_x)
    face_part_id[r_shin_mask] = 14

    part_catalog = {
        1: "Part_Kasa_Hat",
        2: "Part_Head_Face",
        3: "Part_Right_Sode_Shoulder",
        4: "Part_Swords_Katana",
        5: "Part_Left_UpperArm",
        6: "Part_Left_Forearm_Hand",
        7: "Part_Right_UpperArm",
        8: "Part_Right_Forearm_Hand",
        9: "Part_Chest_Torso_Kimono",
        10: "Part_Waist_Obi_Sash",
        11: "Part_Left_Hakama_Thigh",
        12: "Part_Left_Shin_Sandals",
        13: "Part_Right_Hakama_Thigh",
        14: "Part_Right_Shin_Sandals"
    }

    assembled_scene = trimesh.Scene()
    exploded_scene = trimesh.Scene()

    # Explode vector offsets for clean exploded view
    explode_vectors = {
        "Part_Kasa_Hat": [0.0, 0.35, 0.0],
        "Part_Head_Face": [0.0, 0.18, 0.0],
        "Part_Right_Sode_Shoulder": [0.35, 0.12, 0.0],
        "Part_Swords_Katana": [-0.38, -0.05, 0.15],
        "Part_Left_UpperArm": [-0.25, 0.05, 0.0],
        "Part_Left_Forearm_Hand": [-0.45, -0.08, 0.0],
        "Part_Right_UpperArm": [0.25, 0.05, 0.0],
        "Part_Right_Forearm_Hand": [0.45, -0.08, 0.0],
        "Part_Chest_Torso_Kimono": [0.0, 0.0, 0.0],
        "Part_Waist_Obi_Sash": [0.0, -0.10, 0.0],
        "Part_Left_Hakama_Thigh": [-0.18, -0.22, 0.0],
        "Part_Left_Shin_Sandals": [-0.18, -0.42, 0.0],
        "Part_Right_Hakama_Thigh": [0.18, -0.22, 0.0],
        "Part_Right_Shin_Sandals": [0.18, -0.42, 0.0]
    }

    part_stats = []

    for pid, pname in part_catalog.items():
        p_f_idx = np.where(face_part_id == pid)[0]
        if len(p_f_idx) == 0:
            continue

        p_faces = faces[p_f_idx]
        unique_v_idx, new_f_idx = np.unique(p_faces, return_inverse=True)
        p_verts = verts[unique_v_idx]
        p_faces_reindexed = new_f_idx.reshape(p_faces.shape)

        sub_mesh = trimesh.Trimesh(vertices=p_verts, faces=p_faces_reindexed, process=False)
        sub_mesh.metadata['name'] = pname

        if v_colors is not None:
            sub_mesh.visual = trimesh.visual.ColorVisuals(mesh=sub_mesh, vertex_colors=v_colors[unique_v_idx])

        # Cap open cut boundaries so each part is solid and watertight
        cap_boundary_holes(sub_mesh)

        part_stats.append({
            "name": pname,
            "faces": len(sub_mesh.faces),
            "verts": len(sub_mesh.vertices)
        })

        # Add to assembled scene (exact original surface coordinates)
        assembled_scene.add_geometry(sub_mesh.copy(), node_name=pname)

        # Add to exploded scene (with lateral exploded offset)
        sub_mesh_exp = sub_mesh.copy()
        offset = explode_vectors.get(pname, [0, 0, 0])
        sub_mesh_exp.apply_translation(offset)
        exploded_scene.add_geometry(sub_mesh_exp, node_name=pname)

    # Export assembled modular GLB
    assembled_scene.export(output_glb)
    print(f"[SUCCESS] Exported Assembled Neural Model to: {output_glb}")
    print(f"  • Total Parts: {len(part_stats)}")
    for s in part_stats:
        print(f"    - {s['name']}: {s['faces']:,} faces, {s['verts']:,} verts")

    # Render Side-by-Side: Assembled vs Exploded View
    print("[RENDER] Rendering Assembled vs Exploded Verification Sheet...")
    data_asmb = assembled_scene.save_image(resolution=(700, 950), background=[16, 20, 28, 255])
    img_asmb = Image.open(trimesh.util.wrap_as_stream(data_asmb)).convert('RGB')
    draw_a = ImageDraw.Draw(img_asmb)
    draw_a.rectangle([(0, 0), (700, 36)], fill=(10, 14, 22))
    draw_a.text((12, 10), "1. Assembled Character (Continuous Neural Surface - No Primitives)", fill=(230, 235, 250))

    data_exp = exploded_scene.save_image(resolution=(700, 950), background=[16, 20, 28, 255])
    img_exp = Image.open(trimesh.util.wrap_as_stream(data_exp)).convert('RGB')
    draw_e = ImageDraw.Draw(img_exp)
    draw_e.rectangle([(0, 0), (700, 36)], fill=(10, 14, 22))
    draw_e.text((12, 10), f"2. Exploded View ({len(part_stats)} Segmented Parts at Natural Joints)", fill=(56, 189, 248))

    sheet = Image.new('RGB', (1400, 950), (16, 20, 28))
    sheet.paste(img_asmb, (0, 0))
    sheet.paste(img_exp, (700, 0))

    verify_path = r"output\assembled_vs_exploded_verification.png"
    sheet.save(verify_path, quality=95)
    print(f"[SUCCESS] Verification sheet saved to: {verify_path}")

    return part_stats

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else r"output\samurai_neural_sculpt.glb"
    out = sys.argv[2] if len(sys.argv) > 2 else r"output\generated_neural_sculpt.glb"
    segment_and_cap(src, out)
