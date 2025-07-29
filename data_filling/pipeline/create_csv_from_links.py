# data_filling/pipeline/create_csv_from_links.py

import os
import json
import pandas as pd
import uuid
from data_filling.model.multi_input_gptmodel import GPTMultiColumnModel
from data_filling.pipeline.tools_pipeline.extract_framings import extract_all_framings
from data_filling.pipeline.tools_pipeline.utils import ensure_dir, normalize_filename, find_brand_knowledge_path
from data_filling.pipeline.tools_pipeline.download_video_from_url import download_video, clean_folder_if_needed
from data_filling.model.agent.brand_knowledge_agent import BrandKnowledgeAgent

def process_from_links(conf: dict):
    """
    Pipeline pour traiter un CSV contenant des URLs de vidéos.
    """
    input_csv_path = conf["media_csv_path"]
    url_col = conf["media_url_column"]
    brand_col = conf["brand_column"]

    output_dir = conf["output_dir"]
    brands_knowledge_dir = conf["brands_knowledge_dir"]
    download_dir = os.path.join(output_dir, "downloaded_videos")
    ensure_dir(download_dir)

    model = GPTMultiColumnModel(conf)
    agent = BrandKnowledgeAgent(conf)

    df = pd.read_csv(input_csv_path)
    results = []

    for i, row in df.iterrows():
        url = str(row.get(url_col, "")).strip()
        brand = str(row.get(brand_col, "")).strip()
        unique_id = str(uuid.uuid4())
        video_path = os.path.join(download_dir, f"{unique_id}.mp4")

        if not url:
            print(f"❌ No URL found in row {i}, skipping...")
            continue

        print(f"\n⬇️ Downloading video {i+1}/{len(df)}: {url}")
        try:
            download_video(url, video_path)
        except Exception as e:
            print(f"❌ Failed to download video: {e}")
            continue

        # Extract frames & audio
        video_id, frame_paths_by_method = extract_all_framings(video_path, output_dir)

        # Brand knowledge
        brand_knowledge_path = None
        if brand:
            brand_knowledge_path = find_brand_knowledge_path(brand, brands_knowledge_dir)
            if not brand_knowledge_path:
                print(f"⚠️ No knowledge file for '{brand}', generating one...")
                brand_info = agent.generate_knowledge(brand)
                if brand_info:
                    filename = normalize_filename(brand) + ".json"
                    save_path = os.path.join(brands_knowledge_dir, filename)
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(brand_info, f, indent=2, ensure_ascii=False)
                    brand_knowledge_path = save_path

        # Predict
        print(f"🚀 Running model on: {video_id} for brand: {brand or 'No brand'}")
        result_dict = model.predict(frame_paths_by_method, brand_knowledge_path=brand_knowledge_path)

        # Remap keys
        with open(conf["template_path"], "r", encoding="utf-8") as f:
            template = json.load(f)
        key_map = {v["key"]: k for k, v in template.items()}
        remapped_result = {key_map.get(k, k): v for k, v in result_dict.items()}
        remapped_result.update({"video_id": video_id, "video_url": url, "brand": brand})

        results.append(remapped_result)

        clean_folder_if_needed(os.path.join(output_dir, "extracted_frames", video_id))

    # Export CSV
    output_csv = os.path.join(output_dir, "com_case_poc_test.csv")
    df_out = pd.DataFrame(results)
    ordered_columns = ["video_id", "video_url", "brand"] + list(template.keys())
    df_out = df_out.reindex(columns=ordered_columns)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ Final results saved to: {output_csv}")
