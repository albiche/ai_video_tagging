import os
import json
from openai import OpenAI
import httpx


class BrandKnowledgeAgent:
    def __init__(self, config: dict):
        self.output_dir = config.get("brands_knowledge_dir", "config/brand_knowledge")
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = config.get("openai_model_knowledge", "gpt-4o-search-preview-2025-03-11")
        self.client = self._build_client(config)

    def _build_client(self, config: dict):
        api_key = config.get("openai_api_key")
        verify_ssl = config.get("verify_ssl", True)

        if not api_key:
            raise ValueError("Missing OpenAI API key in config.")

        if not verify_ssl:
            print("⚠️ SSL verification disabled (dev mode).")
            http_client = httpx.Client(verify=False)
            return OpenAI(api_key=api_key, http_client=http_client)

        return OpenAI(api_key=api_key)

    def generate_knowledge(self, brand_name: str) -> dict:
        prompt = (
            f"Generate a valid JSON object with the following keys for the brand '{brand_name}':\n"
            "- brand_name: A sentence like 'The brand name is <BrandName>.'\n"
            "- brand_colors: A sentence like 'The main colors of the <BrandName> brand are ...'\n"
            "- brand_elements: A short description of the visual elements typically associated with the brand "
            "(e.g. logo, label, shapes, common visual ingredients...)\n"
            "Only return the JSON with those 3 keys. Do not add explanations or any other text."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a brand analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
        except Exception as e:
            print(f"❌ OpenAI API error during brand knowledge generation: {e}")
            return {}

        try:
            content = response.choices[0].message.content.strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            brand_info = json.loads(content[json_start:json_end])
        except Exception as e:
            print(f"❌ Failed to parse JSON from OpenAI response: {e}")
            return {}

        try:
            brand_info["brand_key"] = brand_name
            brand_name_norm = brand_name.lower().replace(" ", "_")

            file_path = os.path.join(self.output_dir, f"{brand_name_norm}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(brand_info, f, indent=2, ensure_ascii=False)
            print(f"✅ Brand knowledge saved: {file_path}")
        except Exception as e:
            print(f"❌ Failed to save brand knowledge: {e}")
            return {}

        return brand_info
