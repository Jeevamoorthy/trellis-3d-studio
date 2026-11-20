"""
TRELLIS 3D Studio - Universal Desktop Application Backend
Handles local file serving, real image upload & caching, natural language prompt handling,
Hugging Face model weights status, and PyWebView desktop window.
"""

import os
import sys
import json
import base64
import struct
import math
import hashlib
import threading
import urllib.request
import http.server
import socketserver
import subprocess
import time
import webview
from thinking_brain import generate_design_blueprint

# Path handling for script and PyInstaller --onefile
if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    if os.path.exists(os.path.join(exe_dir, "TRELLIS-for-windows")):
        APP_DIR = exe_dir
    elif os.path.exists(os.path.join(os.path.dirname(exe_dir), "TRELLIS-for-windows")):
        APP_DIR = os.path.dirname(exe_dir)
    else:
        APP_DIR = exe_dir
    BUNDLE_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = APP_DIR

OUTPUT_DIR = os.path.join(APP_DIR, "output")
CACHE_DIR = os.path.join(APP_DIR, "cache")
WEIGHTS_DIR = os.path.join(APP_DIR, "TRELLIS-for-windows", "weights", "JeffreyXiang", "TRELLIS-image-large")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
ICON_PATH = os.path.join(OUTPUT_DIR, "app_logo.ico")

def get_weights_status(engine="hunyuan2"):
    if engine in ["hunyuan2", "hunyuan1"]:
        hunyuan_file = os.path.expanduser(r"~/.cache/hy3dgen/tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.safetensors")
        hunyuan_expected_mb = 4700.0
        hunyuan_mb = 0.0
        hunyuan_ready = False
        if os.path.exists(hunyuan_file):
            try:
                hunyuan_mb = round(os.path.getsize(hunyuan_file) / (1024 * 1024), 1)
                hunyuan_ready = hunyuan_mb >= 4600.0
            except Exception:
                pass
        percent = min(100, int((hunyuan_mb / hunyuan_expected_mb) * 100))
        return {
            "status": "ok",
            "engine": engine,
            "is_ready": hunyuan_ready,
            "trellis_ready": hunyuan_ready,
            "dinov2_ready": True,
            "downloaded_mb": hunyuan_mb,
            "total_mb": hunyuan_expected_mb,
            "percent": percent
        }

    total_expected_mb = 4596  # 3,466 MB (TRELLIS FP16) + 1,130 MB (DINOv2 ViT-L/14)
    trellis_mb = 0
    weights_parent = os.path.join(APP_DIR, "TRELLIS-for-windows", "weights")
    if os.path.exists(weights_parent):
        for root, dirs, files in os.walk(weights_parent):
            for f in files:
                try:
                    trellis_mb += os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
                except Exception:
                    pass

    dinov2_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    dinov2_mb = 0
    dinov2_ready = False
    if os.path.exists(dinov2_path):
        dinov2_mb = os.path.getsize(dinov2_path) / (1024 * 1024)
        if dinov2_mb >= 1000:
            dinov2_ready = True

    total_mb = round(trellis_mb + dinov2_mb, 1)

    target_safetensor = os.path.join(WEIGHTS_DIR, "ckpts", "ss_flow_img_dit_L_16l8_fp16.safetensors")
    trellis_ready = False
    if os.path.exists(target_safetensor) and os.path.getsize(target_safetensor) >= 1000 * 1024 * 1024:
        trellis_ready = True

    is_ready = trellis_ready and dinov2_ready
    percent = min(100, int((total_mb / total_expected_mb) * 100))

    return {
        "status": "ok",
        "engine": "trellis",
        "is_ready": is_ready,
        "trellis_ready": trellis_ready,
        "dinov2_ready": dinov2_ready,
        "downloaded_mb": total_mb,
        "total_mb": total_expected_mb,
        "percent": percent
    }

# --- 1. LOCAL HTTP STATIC FILE SERVER ---
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=APP_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith('/api/weights_status'):
            eng = 'trellis' if 'engine=trellis' in self.path else 'hunyuan2'
            status = get_weights_status(engine=eng)
            data = json.dumps(status).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def translate_path(self, path):
        clean_path = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        p_app = os.path.join(APP_DIR, clean_path)
        if os.path.exists(p_app):
            return p_app
        p_bundle = os.path.join(BUNDLE_DIR, clean_path)
        if os.path.exists(p_bundle):
            return p_bundle
        return p_app

    def log_message(self, format, *args):
        pass

def start_local_server():
    server = socketserver.TCPServer(("127.0.0.1", 0), QuietHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return port

# --- 2. PYTHON DESKTOP BRIDGE FOR JAVASCRIPT ---
class StudioBridge:
    def __init__(self, port):
        self.port = port
        self.current_model = None
        self.uploaded_image_path = None
        self.uploaded_views = {}
        self.last_prompt = ""
        self.active_engine = "trellis"
        self.gen_status = {"status": "idle", "stage": "Idle", "percent": 0, "message": "", "model_url": None}

    def upload_reference_image(self, data_url, view_slot="front"):
        """Receives pasted/dragged base64 image data URL and saves to cache."""
        try:
            if not data_url or not isinstance(data_url, str):
                return {"status": "error", "message": "Invalid image data"}
            if "," in data_url:
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            img_bytes = base64.b64decode(encoded)
            target_path = os.path.join(CACHE_DIR, f"input_{view_slot}.png")
            with open(target_path, "wb") as f:
                f.write(img_bytes)
            self.uploaded_views[view_slot] = os.path.abspath(target_path)
            if view_slot == "front" or not self.uploaded_image_path:
                self.uploaded_image_path = self.uploaded_views[view_slot]
            print(f"[IMAGE UPLOAD] Reference image saved for slot '{view_slot}': {target_path} ({len(img_bytes)} bytes)")
            return {"status": "ok", "slot": view_slot, "path": target_path}
        except Exception as e:
            print(f"[IMAGE UPLOAD ERROR] {e}")
            return {"status": "error", "message": str(e)}

    def check_weights_status(self, engine=None):
        """Checks actual weights download status on disk for active engine."""
        return get_weights_status(engine=engine or self.active_engine)

    def get_generation_status(self):
        """Returns live generation progress for UI polling."""
        return self.gen_status

    def plan_with_thinking_brain(self, prompt, category="character"):
        """Invokes Thinking Brain Character Design Director and returns JSON blueprint."""
        try:
            bp = generate_design_blueprint(prompt, category)
            with open(os.path.join(OUTPUT_DIR, "asset_design_blueprint.json"), "w") as f:
                json.dump(bp, f, indent=2)
            with open(os.path.join(CACHE_DIR, "last_blueprint.json"), "w") as f:
                json.dump(bp, f, indent=2)
            return {"status": "ok", "blueprint": bp}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _execute_inference_thread(self, prompt, engine, category, rig):
        try:
            self.active_engine = engine or "trellis"
            engine_name = "Microsoft TRELLIS (1M Poly)" if self.active_engine == "trellis" else ("Tencent Hunyuan3D-1.0 (PBR)" if self.active_engine == "hunyuan1" else "Tencent Hunyuan3D-2.0 (Ultra-HD)")
            self.gen_status = {"status": "running", "stage": f"Stage 1/4: {engine_name} Vision Conditioning", "percent": 20, "message": f"Conditioning features on RTX 4060 CUDA [{engine_name}]...", "model_url": None}
            
            # If user provided a text prompt without custom images, run Thinking Brain
            has_user_image = bool(self.uploaded_views.get("front") or self.uploaded_image_path)
            if prompt and not has_user_image:
                print(f"[THINKING BRAIN] Directing asset design for prompt: '{prompt}' (Category: {category})")
                blueprint = generate_design_blueprint(prompt, category)
                with open(os.path.join(OUTPUT_DIR, "asset_design_blueprint.json"), "w") as f:
                    json.dump(blueprint, f, indent=2)
                with open(os.path.join(CACHE_DIR, "last_blueprint.json"), "w") as f:
                    json.dump(blueprint, f, indent=2)

            # Select reference images (front and side profile)
            front_path = self.uploaded_views.get("front") or self.uploaded_image_path
            if not front_path or not os.path.exists(front_path):
                if os.path.exists(os.path.join(CACHE_DIR, "input_front.png")):
                    front_path = os.path.join(CACHE_DIR, "input_front.png")
                elif os.path.exists(os.path.join(CACHE_DIR, "input_reference.png")):
                    front_path = os.path.join(CACHE_DIR, "input_reference.png")
                else:
                    front_path = os.path.join(APP_DIR, "TRELLIS-for-windows", "assets", "example_image", "T.png")
            
            side_path = self.uploaded_views.get("side") or os.path.join(CACHE_DIR, "input_side.png")
            if not os.path.exists(side_path):
                side_path = ""
            back_path = self.uploaded_views.get("back") or os.path.join(CACHE_DIR, "input_back.png")
            if not os.path.exists(back_path):
                back_path = ""
                
            print(f"[INFERENCE] Starting generation with engine: {self.active_engine}, front: {front_path}, side: {side_path}, back: {back_path}")
            
            venv_python = os.path.join(APP_DIR, "TRELLIS-for-windows", ".venv", "Scripts", "python.exe")
            out_glb = os.path.join(OUTPUT_DIR, "generated_neural_sculpt.glb")
            
            if self.active_engine in ["hunyuan2", "hunyuan1"]:
                pipeline_class = "Hunyuan3DDiTFlowMatchingPipeline"
                checkpoint = "tencent/Hunyuan3D-2mv" if self.active_engine == "hunyuan2" else "tencent/Hunyuan3D-2"
                script_path = os.path.join(APP_DIR, "run_hunyuan3d_generation.py")
                cmd = [
                    venv_python, script_path, front_path, out_glb,
                    "--octree_resolution", "384" if self.active_engine == "hunyuan2" else "256",
                    "--steps", "50",
                    "--guidance_scale", "5.5"
                ]
                if side_path:
                    cmd.extend(["--side_image", side_path])
                if back_path:
                    cmd.extend(["--back_image", back_path])
            else:
                pipeline_class = "TrellisImageTo3DPipeline"
                checkpoint = "JeffreyXiang/TRELLIS-image-large"
                script_path = os.path.join(APP_DIR, "run_trellis_sequential.py")
                cmd = [venv_python, script_path, front_path, out_glb]
                if side_path:
                    cmd.append(side_path)
                if back_path:
                    cmd.append(back_path)

            print(f"[ENGINE DISPATCH] Active Engine: {self.active_engine}")
            print(f"  • Pipeline Class: {pipeline_class}")
            print(f"  • Checkpoint: {checkpoint}")
            print(f"  • Script: {script_path}")
            print(f"  • Views: Front={os.path.basename(front_path)}, Side={bool(side_path)}, Back={bool(back_path)}")

            self.gen_status = {
                "status": "running",
                "stage": f"Running {pipeline_class}",
                "percent": 45,
                "message": f"Generating with {pipeline_class} ({checkpoint}) on RTX 4060...",
                "model_url": None
            }

            if os.path.exists(venv_python) and os.path.exists(script_path):
                res = subprocess.run(cmd, capture_output=True, text=True)
                print(f"[INFERENCE] Return code: {res.returncode}")
                if res.returncode != 0:
                    print(f"[INFERENCE ERROR] {res.stderr}")
                else:
                    print(f"[INFERENCE OUTPUT] {res.stdout[-300:]}")
            
            if os.path.exists(out_glb):
                self.current_model = out_glb
                self.gen_status = {
                    "status": "completed",
                    "stage": "Completed",
                    "percent": 100,
                    "message": f"{pipeline_class} [{checkpoint}] generation complete!",
                    "model_url": f"/output/generated_neural_sculpt.glb?t={int(time.time())}"
                }
            else:
                self.gen_status = {"status": "error", "stage": "Error", "percent": 0, "message": "Failed to output GLB file.", "model_url": None}
        except Exception as e:
            self.gen_status = {"status": "error", "stage": "Error", "percent": 0, "message": str(e), "model_url": None}

    def generate_3d_model(self, prompt, engine, category, rig):
        """Real generation trigger - receives actual prompt and uploaded image."""
        self.last_prompt = prompt
        manifest = {
            "prompt": prompt,
            "engine": engine,
            "category": category,
            "rig": rig,
            "image": self.uploaded_image_path
        }
        with open(os.path.join(CACHE_DIR, "job_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        # Check weights
        status = self.check_weights_status()
        if not status["is_ready"]:
            return {
                "status": "downloading_weights",
                "message": f"TRELLIS Model Weights are currently downloading ({status['downloaded_mb']} MB / {status['total_mb']} MB).",
                "percent": status["percent"]
            }

        t = threading.Thread(target=self._execute_inference_thread, args=(prompt, engine, category, rig), daemon=True)
        t.start()
        return {
            "status": "generating",
            "message": "Generating 3D model using Sequential TRELLIS pipeline on RTX 4060 CUDA..."
        }

    def search_duckduckgo(self, query):
        """Scrapes and caches web reference thumbnails."""
        results = []
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=6):
                    img_url = r.get("image") or r.get("thumbnail")
                    thumb_url = r.get("thumbnail") or img_url
                    h = hashlib.md5(img_url.encode()).hexdigest()[:8]
                    local_fname = f"thumb_{h}.jpg"
                    local_path = os.path.join(CACHE_DIR, local_fname)
                    
                    if not os.path.exists(local_path):
                        try:
                            req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req, timeout=4) as resp, open(local_path, "wb") as f:
                                f.write(resp.read())
                        except Exception:
                            pass
                    
                    if os.path.exists(local_path):
                        results.append({
                            "title": r.get("title", "Reference")[:30],
                            "thumb": f"/cache/{local_fname}",
                            "full": img_url
                        })
        except Exception:
            pass

        return {"status": "ok", "results": results}

    def convert_format(self, target_format, custom_filename=None):
        """Converts model to .obj, .stl, .ply, .glb with full mesh vertices and faces."""
        model_path = self.current_model
        if not model_path or not os.path.exists(model_path):
            default_glb = os.path.join(OUTPUT_DIR, "generated_neural_sculpt.glb")
            if os.path.exists(default_glb):
                model_path = default_glb
            else:
                return {"status": "error", "message": "No model generated yet"}
        
        fmt = target_format.lower()
        clean_name = custom_filename.strip() if (custom_filename and str(custom_filename).strip()) else "lone_warrior_samurai"
        clean_name = re.sub(r'[^a-zA-Z0-9_\- ]', '_', clean_name)
        out_file = os.path.join(OUTPUT_DIR, f"{clean_name}.{fmt}")
        try:
            if fmt == "glb":
                import shutil
                shutil.copyfile(model_path, out_file)
            else:
                import trimesh
                mesh = trimesh.load(model_path, force='mesh')
                mesh.export(out_file)
            print(f"[EXPORT] Successfully saved {fmt.upper()} to: {out_file}")
            # Highlight exported file in Windows Explorer
            try:
                subprocess.Popen(f'explorer /select,"{os.path.abspath(out_file)}"')
            except Exception:
                pass
            return {"status": "ok", "file": out_file, "filename": os.path.basename(out_file)}
        except Exception as e:
            print(f"[EXPORT ERROR] {e}")
            return {"status": "error", "message": str(e)}

    def reproject_texture(self, y_offset=0.0, scale_factor=1.0):
        """Re-projects the front reference image onto the current 3D mesh with adjusted vertical offset and scale."""
        model_path = self.current_model or os.path.join(OUTPUT_DIR, "generated_neural_sculpt.glb")
        front_img = self.uploaded_views.get("front") or self.uploaded_image_path
        if not front_img or not os.path.exists(front_img):
            alts = [
                os.path.join(OUTPUT_DIR, "samurai_warrior_front.png"),
                os.path.join(CACHE_DIR, "input_front.png"),
                os.path.join(CACHE_DIR, "input_reference.png"),
                os.path.join(CACHE_DIR, "tsunade_input.png")
            ]
            for a in alts:
                if os.path.exists(a):
                    front_img = a
                    break
        if not os.path.exists(model_path) or not os.path.exists(front_img):
            return {"status": "error", "message": "Model or reference image not found"}
        try:
            import trimesh
            from run_hunyuan3d_generation import texture_mesh_from_image
            loaded = trimesh.load(model_path)
            if isinstance(loaded, trimesh.Scene):
                for name, geom in loaded.geometry.items():
                    texture_mesh_from_image(geom, front_img, y_offset=float(y_offset), scale_factor=float(scale_factor))
                loaded.export(model_path)
            else:
                mesh = texture_mesh_from_image(loaded, front_img, y_offset=float(y_offset), scale_factor=float(scale_factor))
                mesh.export(model_path)
            print(f"[REPROJECT] Successfully updated vertex colors on: {model_path}")
            return {"status": "ok", "model_url": f"/output/{os.path.basename(model_path)}?t={int(time.time()*1000)}"}
        except Exception as e:
            print(f"[REPROJECT ERROR] {e}")
            return {"status": "error", "message": str(e)}

    def open_output_folder(self):
        """Opens Windows Explorer in output folder."""
        os.startfile(OUTPUT_DIR)
        return {"status": "ok"}

def main():
    try:
        port = start_local_server()
        bridge = StudioBridge(port)
        app_url = f"http://127.0.0.1:{port}/index.html"
        
        window = webview.create_window(
            title="TRELLIS 3D Studio - Universal AI Model Studio (RTX 4060 CUDA)",
            url=app_url,
            js_api=bridge,
            width=1380,
            height=860,
            min_size=(1080, 700),
            resizable=True
        )
        webview.start(icon=ICON_PATH if os.path.exists(ICON_PATH) else None)
    except Exception as e:
        with open(os.path.join(APP_DIR, "studio_error.log"), "w") as ef:
            ef.write(str(e))

if __name__ == "__main__":
    main()
