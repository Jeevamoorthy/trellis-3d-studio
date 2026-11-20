import os
import sys
import json
import urllib.request
import re

NIM_CONFIG_PATH = r"C:\Users\JEEVA\.gemini\antigravity\nim\nim_config.json"

DIRECTOR_SYSTEM_PROMPT = """You are a character design director for a 3D pipeline. The user gives a short idea. Output ONLY valid JSON with these fields:

{
  "name": "",
  "body": {"gender_presentation":"", "height_cm":0, "build":"", "age_range":""},
  "pose": "neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat, facing camera",
  "head": {"face_shape":"", "eyes":"", "expression":"neutral", "hair": {"style":"", "length":"", "color":""}},
  "outfit_layers": [
    {"part":"", "layer_order":1, "material":"", "fit":"tight|loose", "color":"", "cloth_sim_needed": true}
  ],
  "parts_to_generate": ["body","head","hair","top","bottom","coat","shoes"],
  "avoid": ["wind-blown hair","long trailing capes","props in hands", "kneeling or action poses","overlapping limbs"],
  "image_prompt_front": "",
  "image_prompt_side": "",
  "image_prompt_back": ""
}

Rules:
- Always design for rigging: neutral A-pose, limbs separated from the torso, clear silhouette, no limb hidden by clothing or hair.
- Keep clothing as separate layers with distinct boundaries.
- Prefer simple, readable shapes over fine ornament.
- Write each image prompt so the same character appears in all three views: same outfit, proportions and colors, plain white background, full body, centered, soft even lighting, no shadows.
Format:
Full-body character reference, [front/side/back] view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. [Character description from JSON]. Clean semi-realistic anime style, crisp linework, readable fabric folds and hair strand groups. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.

For inanimate objects (bookshelf, tree, leaf, book, vehicle) or animals/creatures:
- Adapt the schema logically: for creatures use neutral anatomical stance with all limbs separated for rigging; for inanimate objects use clean upright orthographic elevation and profile, layered parts, no floor clutter.
"""

def generate_design_blueprint(idea_prompt, category="character"):
    """
    Invokes the Character & Asset Design Director Thinking Brain to produce
    rigging-optimized JSON and 3-view orthographic prompts.
    """
    # 1. Try NIM Gateway / NVIDIA API if key exists
    api_key = None
    if os.path.exists(NIM_CONFIG_PATH):
        try:
            with open(NIM_CONFIG_PATH, "r") as f:
                conf = json.load(f)
                api_key = conf.get("api_key")
        except Exception:
            pass

    # First attempt: Local NIM router
    endpoints = [
        ("http://127.0.0.1:8000/v1/chat/completions", None, "architect"),
        ("https://integrate.api.nvidia.com/v1/chat/completions", api_key, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
        ("https://integrate.api.nvidia.com/v1/chat/completions", api_key, "nvidia/llama-3.1-nemotron-70b-instruct")
    ]

    user_msg = f"Design idea: '{idea_prompt}'. Asset Category: {category}."

    for url, key, model in endpoints:
        if "integrate.api.nvidia.com" in url and not key:
            continue
        try:
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.2,
                "max_tokens": 1200
            }

            timeout_sec = 1.5 if "127.0.0.1" in url else 6.0
            req = urllib.request.Request(url, headers=headers, data=json.dumps(payload).encode("utf-8"))
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                
                # Extract JSON block
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    print(f"[THINKING BRAIN] Successfully planned design for: {parsed.get('name')}")
                    return parsed
        except Exception as e:
            # Fall through to next endpoint or deterministic architect
            continue

    # Fallback deterministic Character Design Director Planner
    print(f"[THINKING BRAIN] Using built-in Character Design Director Planner for: {idea_prompt} (Category: {category})")
    clean_name = idea_prompt.strip().split('.')[0].capitalize()[:60]
    
    is_char = (category.lower() == "character") or any(k in idea_prompt.lower() for k in ["warrior", "samurai", "man", "woman", "girl", "boy", "knight", "human", "character", "tsunade", "ninja", "fighter"])
    is_creature = not is_char and ((category.lower() == "creature") or any(k in idea_prompt.lower() for k in ["animal", "cat", "dog", "wolf", "dragon", "beast", "creature", "monster"]))
    is_prop = not is_char and not is_creature

    if is_char:
        # Humanoid Character (e.g. samurai / warrior / anime character)
        blueprint = {
            "name": "Lone Warrior Samurai",
            "body": {
                "gender_presentation": "male",
                "height_cm": 182,
                "build": "lean, athletic, broad-shouldered",
                "age_range": "experienced adult"
            },
            "pose": "neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat, facing camera",
            "head": {
                "face_shape": "chiseled jaw shrouded in deep shadow",
                "eyes": "focused intense gaze",
                "expression": "stoic neutral",
                "hair": {
                  "style": "tied samurai topknot hidden beneath wide conical straw hat kasa with radial ribbing",
                  "length": "medium",
                  "color": "jet black"
                }
            },
            "outfit_layers": [
                {"part": "wide_conical_kasa_hat", "layer_order": 1, "material": "woven straw with radial ribbing", "fit": "resting low over brow", "color": "weathered straw tan/charcoal", "cloth_sim_needed": False},
                {"part": "inner_kimono", "layer_order": 2, "material": "charcoal-black coarse linen", "fit": "loose-fitting, wide sleeves tapering at wrists", "color": "charcoal black", "cloth_sim_needed": True},
                {"part": "chest_strap", "layer_order": 3, "material": "braided cord", "fit": "diagonal across chest", "color": "dark crimson red", "cloth_sim_needed": False},
                {"part": "right_sode_shoulder_guard", "layer_order": 4, "material": "asymmetrical lacquered iron plates with silk cord binding", "fit": "fitted over right deltoid", "color": "matte black with dark red lacing", "cloth_sim_needed": False},
                {"part": "thick_obi_sash", "layer_order": 5, "material": "heavy folded silk sash", "fit": "tight waist wrap", "color": "black with dark red accents", "cloth_sim_needed": False},
                {"part": "dual_sheathed_swords", "layer_order": 6, "material": "lacquered wood scabbards with ray-skin tsuka hilts", "fit": "katana and wakizashi secured at left hip angled downward", "color": "black lacquer with crimson sageo cord", "cloth_sim_needed": False},
                {"part": "wide_hakama_trousers", "layer_order": 7, "material": "heavy structured pleated cotton", "fit": "wide-pleated, deep vertical folds flaring toward ankles", "color": "deep charcoal black", "cloth_sim_needed": True},
                {"part": "footwear", "layer_order": 8, "material": "woven straw waraji sandals and tabi socks", "fit": "laced around calves", "color": "natural straw and black cloth", "cloth_sim_needed": False}
            ],
            "parts_to_generate": ["body", "kasa_hat", "head", "kimono_top", "sode_shoulder", "obi_waist", "swords", "hakama_pants", "sandals"],
            "avoid": ["wind-blown hair", "long trailing capes", "props in hands", "kneeling or action poses", "overlapping limbs"],
            "image_prompt_front": "Full-body character reference, front view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. Lone warrior samurai, lean athletic broad-shouldered build, wide conical straw hat kasa with radial ribbing, charcoal-black kimono with wide tapered sleeves, asymmetrical dark segmented sode shoulder guard on right shoulder, diagonal dark crimson chest strap, heavy wide-pleated black hakama trousers flaring to ankles, thick obi sash holding sheathed katana and wakizashi at left hip, waraji sandals. Clean semi-realistic anime style, crisp linework, readable fabric folds and sharp armor boundaries. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_side": "Full-body character reference, side profile view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. Lone warrior samurai, lean athletic broad-shouldered build, wide conical straw hat kasa with radial ribbing, charcoal-black kimono with wide tapered sleeves, asymmetrical dark segmented sode shoulder guard on right shoulder, diagonal dark crimson chest strap, heavy wide-pleated black hakama trousers flaring to ankles, thick obi sash holding sheathed katana and wakizashi at left hip, waraji sandals. Clean semi-realistic anime style, crisp linework, readable fabric folds and sharp armor boundaries. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_back": "Full-body character reference, back view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. Lone warrior samurai, lean athletic broad-shouldered build, wide conical straw hat kasa with radial ribbing, charcoal-black kimono with wide tapered sleeves, asymmetrical dark segmented sode shoulder guard on right shoulder, diagonal dark crimson chest strap, heavy wide-pleated black hakama trousers flaring to ankles, thick obi sash holding sheathed katana and wakizashi at left hip, waraji sandals. Clean semi-realistic anime style, crisp linework, readable fabric folds and sharp armor boundaries. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting."
        }
    elif is_creature:
        blueprint = {
            "name": clean_name,
            "body": {"gender_presentation": "inanimate_object", "height_cm": 180, "build": "structured architectural", "age_range": "timeless"},
            "pose": "canonical upright orthographic elevation, centered, no tilt, level ground alignment",
            "head": {"face_shape": "geometric frame", "eyes": "none", "expression": "neutral", "hair": {"style": "none", "length": "none", "color": "none"}},
            "outfit_layers": [
                {"part": "primary_structure", "layer_order": 1, "material": "matte hardwood / timber", "fit": "rigid", "color": "natural umber", "cloth_sim_needed": False},
                {"part": "inner_shelves_and_panels", "layer_order": 2, "material": "sanded timber", "fit": "fitted", "color": "warm wood grain", "cloth_sim_needed": False},
                {"part": "contained_items_and_detail", "layer_order": 3, "material": "cloth and leather bindings", "fit": "organized", "color": "assorted muted tones", "cloth_sim_needed": False}
            ],
            "parts_to_generate": ["frame", "shelves", "panels", "detail_elements"],
            "avoid": ["ground clutter", "excessive tilt", "occluded back planes", "unsupported floating geometry"],
            "image_prompt_front": f"Full asset reference, front elevation view, single object, centered upright canonical angle. {clean_name}. Clean semi-realistic architectural style, crisp linework, readable seams and distinct material boundaries. Entire asset visible top to bottom, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_side": f"Full asset reference, side profile view, single object, centered upright canonical angle. {clean_name}. Clean semi-realistic architectural style, crisp linework, readable seams and distinct material boundaries. Entire asset visible top to bottom, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_back": f"Full asset reference, back elevation view, single object, centered upright canonical angle. {clean_name}. Clean semi-realistic architectural style, crisp linework, readable seams and distinct material boundaries. Entire asset visible top to bottom, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting."
        }
    elif is_creature:
        blueprint = {
            "name": clean_name,
            "body": {"gender_presentation": "creature", "height_cm": 110, "build": "athletic anatomical quadruped", "age_range": "prime adult"},
            "pose": "neutral anatomical quadruped A-pose, all four limbs planted symmetrically and separated from torso, tail relaxed in neutral curve, neck extended, facing camera",
            "head": {"face_shape": "defined snout and brow", "eyes": "forward-facing alert", "expression": "neutral", "hair": {"style": "short dense pelt", "length": "short", "color": "tawny grey"}},
            "outfit_layers": [
                {"part": "base_musculature", "layer_order": 1, "material": "dense fur pelt", "fit": "anatomical", "color": "natural gradient", "cloth_sim_needed": False},
                {"part": "crest_or_mane", "layer_order": 2, "material": "coarse guard hair", "fit": "layered", "color": "dark slate", "cloth_sim_needed": False}
            ],
            "parts_to_generate": ["torso", "head", "neck", "forelimbs", "hindlimbs", "tail"],
            "avoid": ["crouching or biting poses", "overlapping paws", "tucked tail", "props or harness"],
            "image_prompt_front": f"Full-body creature reference, front view, single creature, neutral quadruped A-pose, limbs separated from body, paws planted flat, facing camera. {clean_name}. Clean semi-realistic anatomy, crisp linework, readable fur clumps and distinct muscle boundaries. Entire body visible head to paws, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_side": f"Full-body creature reference, side view, single creature, neutral quadruped A-pose, limbs separated from body, paws planted flat, facing camera. {clean_name}. Clean semi-realistic anatomy, crisp linework, readable fur clumps and distinct muscle boundaries. Entire body visible head to paws, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_back": f"Full-body creature reference, back view, single creature, neutral quadruped A-pose, limbs separated from body, paws planted flat, facing camera. {clean_name}. Clean semi-realistic anatomy, crisp linework, readable fur clumps and distinct muscle boundaries. Entire body visible head to paws, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting."
        }
    else:
        # Humanoid Character (e.g. anime / game character)
        blueprint = {
            "name": clean_name,
            "body": {"gender_presentation": "female" if "girl" in idea_prompt.lower() or "tsunade" in idea_prompt.lower() else "androgynous/male", "height_cm": 168, "build": "athletic toned", "age_range": "early adult"},
            "pose": "neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat, facing camera",
            "head": {"face_shape": "oval with defined chin", "eyes": "clear expressive anime eyes", "expression": "neutral", "hair": {"style": "structured layered locks", "length": "shoulder length", "color": "blonde" if "tsunade" in idea_prompt.lower() else "dark brown"}},
            "outfit_layers": [
                {"part": "base_suit", "layer_order": 1, "material": "cotton weave", "fit": "tight", "color": "cream white", "cloth_sim_needed": False},
                {"part": "outer_coat_haori", "layer_order": 2, "material": "structured canvas", "fit": "loose", "color": "emerald green" if "tsunade" in idea_prompt.lower() else "slate blue", "cloth_sim_needed": True},
                {"part": "waist_sash_obi", "layer_order": 3, "material": "silk sash", "fit": "tight", "color": "deep indigo", "cloth_sim_needed": False},
                {"part": "pants_shorts", "layer_order": 4, "material": "tailored fabric", "fit": "fitted", "color": "dark slate", "cloth_sim_needed": False},
                {"part": "footwear", "layer_order": 5, "material": "leather sandals/boots", "fit": "fitted", "color": "charcoal black", "cloth_sim_needed": False}
            ],
            "parts_to_generate": ["body", "head", "hair", "top", "bottom", "coat", "shoes"],
            "avoid": ["wind-blown hair", "long trailing capes", "props in hands", "kneeling or action poses", "overlapping limbs"],
            "image_prompt_front": f"Full-body character reference, front view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. {clean_name}. Clean semi-realistic anime style, crisp linework, readable fabric folds and hair strand groups. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_side": f"Full-body character reference, side profile view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. {clean_name}. Clean semi-realistic anime style, crisp linework, readable fabric folds and hair strand groups. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting.",
            "image_prompt_back": f"Full-body character reference, back view, single character, neutral A-pose, arms 30 degrees from body, hands relaxed, feet flat. {clean_name}. Clean semi-realistic anime style, crisp linework, readable fabric folds and hair strand groups. Entire body visible head to toe, centered, orthographic-style, plain white background, no floor, no shadow, no text, soft even studio lighting."
        }

    return blueprint

if __name__ == "__main__":
    test_prompt = sys.argv[1] if len(sys.argv) > 1 else "Tsunade modern cel-shaded anime"
    res = generate_design_blueprint(test_prompt)
    print(json.dumps(res, indent=2))
