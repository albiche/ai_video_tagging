import os
import re
import glob
import json

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_filename(brand_name: str) -> str:
    return re.sub(r'\W+', '_', brand_name.strip().lower())


def find_brand_knowledge_path(brand_key: str, knowledge_dir: str) -> str | None:
    brand_key_norm = brand_key.strip().lower()
    for json_path in glob.glob(os.path.join(knowledge_dir, "*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                key = data.get("brand_key", "").strip().lower()
                if key == brand_key_norm:
                    print(f"🔍 Found brand knowledge for '{brand_key}' in {json_path}")
                    return json_path
        except Exception as e:
            print(f"⚠️ Error reading brand file '{json_path}': {e}")
    return None