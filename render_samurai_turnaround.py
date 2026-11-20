import os
import trimesh
import numpy as np
from PIL import Image, ImageDraw

def render_samurai_views():
    glb_path = r"output\generated_neural_sculpt.glb"
    mesh = trimesh.load(glb_path, force='mesh')
    verts = mesh.vertices.copy()
    faces = mesh.faces
    normals = mesh.vertex_normals.copy()
    
    # 1. Clay Mode Vertex Colors (#8a8d93 base, warm key, cool fill, soft rim, crease AO)
    base_albedo = np.array([138, 141, 147], dtype=np.float32)
    key_dir = np.array([-0.42, 0.58, 0.69], dtype=np.float32)
    key_dir /= np.linalg.norm(key_dir)
    dot_key = np.maximum(0.0, np.dot(normals, key_dir))
    key_term = np.power(dot_key, 1.15) * 0.72

    fill_dir = np.array([0.65, -0.28, 0.55], dtype=np.float32)
    fill_dir /= np.linalg.norm(fill_dir)
    dot_fill = np.maximum(0.0, np.dot(normals, fill_dir))
    fill_term = np.power(dot_fill, 1.10) * 0.38

    cam_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot_cam = np.abs(np.dot(normals, cam_dir))
    rim_term = np.power(1.0 - dot_cam, 2.6) * 0.26
    hemi_term = np.maximum(0.0, normals[:, 1] * 0.5 + 0.5) * 0.35 + 0.18
    crease_ao = np.power(np.maximum(0.08, dot_cam), 0.50)
    depth_shadow = 0.42 + 0.58 * crease_ao

    half_vec = np.array([-0.28, 0.42, 0.86], dtype=np.float32)
    half_vec /= np.linalg.norm(half_vec)
    dot_h = np.maximum(0.0, np.dot(normals, half_vec))
    matte_sheen = np.power(dot_h, 14.0) * 0.10

    light_r = (hemi_term * 0.98 + key_term * 1.08 + fill_term * 0.88 + rim_term * 0.95) * depth_shadow
    light_g = (hemi_term * 1.00 + key_term * 1.02 + fill_term * 0.96 + rim_term * 0.98) * depth_shadow
    light_b = (hemi_term * 1.04 + key_term * 0.92 + fill_term * 1.10 + rim_term * 1.06) * depth_shadow

    final_r = np.clip(base_albedo[0] * light_r + matte_sheen * 255.0, 38.0, 195.0)
    final_g = np.clip(base_albedo[1] * light_g + matte_sheen * 255.0, 40.0, 198.0)
    final_b = np.clip(base_albedo[2] * light_b + matte_sheen * 255.0, 44.0, 204.0)
    clay_colors = np.stack([final_r, final_g, final_b, np.full_like(final_r, 255.0)], axis=-1).astype(np.uint8)

    angles = [
        ("Front View", 0),
        ("3/4 Profile", 45),
        ("Side View", 90),
        ("Back View", 180)
    ]

    clay_panels = []
    for label, yaw_deg in angles:
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.apply_translation(-m.centroid)
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=clay_colors)

        # Rotate around Y
        rot = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0])
        m.apply_transform(rot)

        scene = m.scene()
        cam_trans = trimesh.transformations.euler_matrix(np.radians(-6), 0, 0, 'sxyz')
        cam_dist = max(m.extents) * 1.65
        cam_trans[:3, 3] = cam_trans[:3, 2] * cam_dist
        scene.camera_transform = cam_trans

        data = scene.save_image(resolution=(500, 750), background=[16, 20, 28, 255])
        img = Image.open(trimesh.util.wrap_as_stream(data)).convert("RGB")

        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (500, 32)], fill=(10, 14, 22))
        draw.text((10, 8), f"Lone Warrior Samurai - {label}", fill=(138, 141, 147))
        clay_panels.append(img)

    sheet = Image.new("RGB", (2000, 750), (16, 20, 28))
    for i, p in enumerate(clay_panels):
        sheet.paste(p, (i * 500, 0))

    out_path = r"output\samurai_clay_turnaround_4views.png"
    sheet.save(out_path, quality=95)
    print(f"[SUCCESS] Samurai Clay Turnaround saved to: {out_path}")

if __name__ == "__main__":
    render_samurai_views()
