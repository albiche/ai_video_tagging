# data_filling/pipeline/process_video.py

import os
import json
from data_filling.model.multi_input_gptmodel import GPTMultiColumnModel
from data_filling.pipeline.tools_pipeline.extract_framings import extract_all_framings
from data_filling.pipeline.tools_pipeline.utils import ensure_dir, normalize_filename, find_brand_knowledge_path
from data_filling.model.agent.brand_knowledge_agent import BrandKnowledgeAgent

def process_all_videos(conf: dict):
    """
    Pipeline pour traiter un dossier de vidéos locales.
    """
    input_video_dir = conf["input_video_dir"]
    output_dir = conf["output_dir"]
    brands_knowledge_dir = conf["brands_knowledge_dir"]

    # Charger le mapping video_id -> brand
    with open(conf["brand_map_path"], "r", encoding="utf-8") as f:
        video_to_brand = json.load(f)

    model = GPTMultiColumnModel(conf)
    agent = BrandKnowledgeAgent(conf)

    video_files = [
        os.path.join(input_video_dir, f)
        for f in os.listdir(input_video_dir)
        if f.lower().endswith((".mp4", ".mov"))
    ]

    for video_path in video_files:
        video_id, frame_paths_by_method = extract_all_framings(video_path, output_dir)
        brand_name = video_to_brand.get(video_id)
        brand_knowledge_path = None

        if brand_name:
            brand_knowledge_path = find_brand_knowledge_path(brand_name, brands_knowledge_dir)
            if not brand_knowledge_path:
                print(f"⚠️ No knowledge file for '{brand_name}', generating one...")
                brand_info = agent.generate_knowledge(brand_name)
                if brand_info:
                    filename = normalize_filename(brand_name) + ".json"
                    save_path = os.path.join(brands_knowledge_dir, filename)
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(brand_info, f, indent=2, ensure_ascii=False)
                    brand_knowledge_path = save_path

        print(f"\n🚀 Running model on video: {video_id} for brand: {brand_name or 'Unknown'}")
        results = model.predict(frame_paths_by_method, brand_knowledge_path=brand_knowledge_path)

        result_path = os.path.join(output_dir, "outputs_arch", f"{video_id}.json")
        ensure_dir(os.path.dirname(result_path))

        with open(result_path, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2, ensure_ascii=False)

        print(f"✅ Saved results to {result_path}")
