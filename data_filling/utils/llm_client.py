def query_llm_with_images(images: list, prompt_json: dict) -> str:
    """
    Envoie les images + JSON de questions à GPT vision (mock pour l’instant)
    """
    # Tu pourrais mettre ici une vraie requête OpenAI vision
    return """{
        "logo_or_ocr_first_5s": "1",
        "logo_first_2s": "1",
        "brandname_ocr_first_2s": "0"
    }"""
