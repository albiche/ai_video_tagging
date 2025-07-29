# main.py

import yaml
from data_filling.pipeline.create_csv_from_links import process_from_links
from data_filling.pipeline.process_video import process_all_videos

if __name__ == "__main__":
    # Charger la configuration
    CONFIG_PATH = "config/conf.yml"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # Choisir la bonne pipeline en fonction des clés présentes
    if "media_csv_path" in conf:
        process_from_links(conf)  # Pipeline CSV -> téléchargement + extraction
    elif "input_video_dir" in conf:
        process_all_videos(conf)  # Pipeline dossier vidéos -> extraction
    else:
        raise ValueError("❌ Aucune source détectée dans la configuration. Ajoute 'media_csv_path' ou 'input_video_dir'.")
