import base64
import cv2
import json
import httpx
from openai import OpenAI
import os
from data_filling.model.tools.prompt_builder import (
    smart_split_prompt,
    merge_responses,
    build_prompt_messages,
)
from typing import List, Dict

from data_filling.model.tools.frame_selector import select_frames
from data_filling.model.tools.batch_grouper import group_tags_by_batch
from data_filling.model.tools.result_parser import parse_gpt_output
from data_filling.model.tools.compute_ratios import compute_frame_ratios
from data_filling.model.tools.audio_selector import select_audio
from data_filling.model.tools.mapper import remap_keys_to_labels

class GPTMultiColumnModel:
    """
    Model for processing structured prompts across multiple frame sets from a video.
    Handles batching, chunking, retries, and GPT parsing using a vision-enabled model (e.g., GPT-4o).
    """

    def __init__(self, config: dict):
        self._config = config
        self._client = self._build_client()
        self._model_name = config.get("openai_model", "gpt-4o")
        self._model_transcript_name = config.get("openai_model_transcript", "gpt-4o-transcribe")
        self._template_path = config.get("template_path")

    def _build_client(self):
        api_key = self._config.get("openai_api_key")
        verify_ssl = self._config.get("verify_ssl", True)
        if not api_key:
            raise ValueError("Missing 'openai_api_key' in config.")
        if not verify_ssl:
            print("⚠️ SSL verification disabled (dev mode).")
            http_client = httpx.Client(verify=False)
            return OpenAI(api_key=api_key, http_client=http_client)
        return OpenAI(api_key=api_key)

    def _load_template(self, brand_knowledge_path: str = None):
        with open(self._template_path, "r", encoding="utf-8") as f:
            template = json.load(f)
        # Si aucun fichier brand_knowledge, on retourne le template brut
        if not brand_knowledge_path or not os.path.exists(brand_knowledge_path):
            return template
        # Charger le fichier de contexte de marque
        try:
            with open(brand_knowledge_path, "r", encoding="utf-8") as f:
                brand_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load brand knowledge from {brand_knowledge_path}: {e}")
            return template

        for col, settings in template.items():
            key = settings.get("prompt_additional")
            if key and key in brand_data:
                additional_info = brand_data[key].strip()
                prompt = settings.get("prompt_ai", "").strip()
                settings["prompt_ai"] = f"Brand context: {additional_info}. Then, {prompt}"

        return template

    def _encode_image(self, img_path: str) -> str:
        image = cv2.imread(img_path)
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError(f"Failed to encode image: {img_path}")
        return base64.b64encode(buffer).decode("utf-8")

    def _encode_audio(self, audio_path: str) -> str:
        """
        Encode an audio file to base64.
        """
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode("utf-8")

    def _send_request(self, base64_images: List[str], transcriptions: List[str], prompt_data: Dict) -> dict:
        messages = build_prompt_messages(prompt_data, base64_images, transcriptions)
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=8000,
            temperature=0
        )
        return self._parse_response(response.choices[0].message.content.strip())

    def _send_request_transcript(self, audio_path: str) -> str:
        with open(audio_path, "rb") as audio_file:
                transcription = self._client.audio.transcriptions.create(
                    model=self._model_transcript_name,  # exemple : "gpt-4o-transcribe"
                    file=audio_file,
                    response_format="text"
                )
        return transcription.strip() if isinstance(transcription, str) else ""


    def _parse_response(self, raw: str) -> dict:
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        try:
            return json.loads(raw[json_start:json_end])
        except Exception as e:
            print(f"⚠️ JSON parsing error: {e}")
            return {}

    def _validate_chunk(self, raw_response, prompt_data):
        validated, invalid = {}, {}

        for key, value in raw_response.items():
            if key not in prompt_data:
                print(f"❌ Unknown key '{key}' in GPT response. Ignored.")
                continue

            val_str = str(value).strip()
            accepted = prompt_data[key].get("accepted_values", [])

            # Always accept N/A
            if val_str == "N/A":
                validated[key] = "N/A"
                continue

            # If accepted is empty or not a list → accept anything
            if not accepted or not isinstance(accepted, list):
                validated[key] = val_str
                continue

            # If accepted values is a simple list like ["yes", "no"] or ["1", "0"]
            if all(isinstance(x, str) for x in accepted) and not any("0-" in x for x in accepted):
                if val_str in accepted:
                    validated[key] = val_str
                else:
                    invalid[key] = val_str
                continue

            # If accepted values contain a "0-100" style entry
            range_strs = [a for a in accepted if "-" in a and a.replace("-", "").isdigit()]
            if range_strs:
                for r in range_strs:
                    try:
                        low, high = map(int, r.split("-"))
                        num = int(val_str)
                        if low <= num <= high:
                            validated[key] = val_str
                            break
                    except ValueError:
                        continue
                else:
                    invalid[key] = val_str
                continue

            # Fallback: accept
            validated[key] = val_str

        return validated, invalid

    def _multi_prompt_process(self, prompt_data, base64_images=None, transcriptions=None, ratios=None,
                              current_frame_method=None):
        all_responses = []
        invalid_fields = []
        frames_per_chunk = []

        if not base64_images and not transcriptions:
            raise ValueError("❌ No images or transcription provided for processing. At least one must be non-empty.")
        print(transcriptions, prompt_data)
        chunks = smart_split_prompt(
            prompt_data=prompt_data,
            images_b64=base64_images or [],
            transcriptions=transcriptions or [],
            max_tokens=8000,
            model=self._model_name,
            max_images_per_chunk=10,
            max_chunks=15,
            split_image=True
        )

        if not chunks:
            print("❌ Aborted: prompt too heavy to split reasonably.")
            return {k: "N/A" for k in prompt_data}

        print(f"🔄 Processing {len(chunks)} initial chunk(s)...")

        for i, (prompt_chunk, image_chunk, transcription_chunk) in enumerate(chunks):
            print(
                f"🧩 Chunk {i + 1}/{len(chunks)} — {len(prompt_chunk)} fields, {len(image_chunk)} image(s), {len(transcription_chunk)} transcription(s)")
            raw = self._send_request(image_chunk, transcription_chunk, prompt_chunk)
            print(raw)

            validated, invalid = self._validate_chunk(raw, prompt_chunk)
            all_responses.append(validated)

            frames_per_chunk.append(len(image_chunk) if image_chunk else 1)  # texte = 1 ratio = 1

            for k in invalid:
                if k not in [key for d in all_responses for key in d]:
                    invalid_fields.append(k)

        # Retry logic
        if invalid_fields:
            print(f"🔁 Retrying {len(invalid_fields)} invalid field(s)...")
            retry_prompt = {k: prompt_data[k] for k in invalid_fields}
            retry_chunks = smart_split_prompt(
                prompt_data=retry_prompt,
                images_b64=base64_images or [],
                transcriptions=transcriptions or [],
                max_tokens=6000,
                model=self._model_name,
                max_images_per_chunk=10,
                max_chunks=10,
                split_image=True
            )

            if not retry_chunks:
                print("⚠️ Retry prompt too heavy, skipping retry.")
            else:
                for i, (prompt_chunk, image_chunk, transcription_chunk) in enumerate(retry_chunks):
                    print(f"🔁 Retry Chunk {i + 1}/{len(retry_chunks)} — {len(prompt_chunk)} fields")
                    raw = self._send_request(image_chunk, transcription_chunk, prompt_chunk)
                    validated, _ = self._validate_chunk(raw, prompt_chunk)
                    all_responses.append(validated)

                    frames_per_chunk.append(len(image_chunk) if image_chunk else 1)

        # Final merge
        merged = merge_responses(
            all_responses,
            batch_config=prompt_data,
            frames_per_chunk=frames_per_chunk,
            ratios=ratios,
            current_frame_method=current_frame_method
        )

        print('merged', merged)

        for k in prompt_data:
            if k not in merged:
                merged[k] = "N/A"

        return merged

    def predict(self, video_frames_dict: dict, brand_knowledge_path: str = None) -> dict:

        print(brand_knowledge_path)
        """
        Main prediction routine, supports brand-specific prompt enrichment.
        """
        template = self._load_template(brand_knowledge_path)
        batches = group_tags_by_batch(template)
        ratios = compute_frame_ratios(video_frames_dict)
        print("template", template)
        final_results = {}

        for (frame_method, frames_used, split_possible, audio_key), batch_config in batches:
            selected_frames = []
            selected_audio_paths = []
            transcriptions = []

            # Try to get frames
            if frame_method and frame_method in video_frames_dict:
                full_frames = video_frames_dict[frame_method]
                selected_frames = select_frames(full_frames, frames_used)
                print(f"📸 Selected {len(selected_frames)} frame(s) for {frame_method}")
            else:
                print(f"⚠️ Missing frames for method: {frame_method}")

            # Try to get audio
            if audio_key and audio_key in video_frames_dict:
                selected_audio_paths = video_frames_dict[audio_key]
                print(f"🎵 Selected {len(selected_audio_paths)} audio file(s) for {audio_key}")
                # Transcribe each audio file
                for audio_path in selected_audio_paths:
                    try:
                        transcription_text = self._send_request_transcript(audio_path)
                        transcriptions.append(transcription_text)
                    except Exception as e:
                        print(f"⚠️ Transcription failed for {audio_path}: {e}")
            else:
                print(f"⚠️ Missing audio for key: {audio_key}")

            # Encode media
            base64_images = [self._encode_image(p) for p in selected_frames] if selected_frames else []

            # Validation
            if not base64_images and not transcriptions:
                print(
                    f"⚠️ Skipping batch: no frames nor audio available for frame_method={frame_method} audio={audio_key}"
                )
                continue

            # Process
            result = self._multi_prompt_process(
                batch_config,
                base64_images=base64_images,
                transcriptions=transcriptions,
                ratios=ratios,
                current_frame_method=frame_method
            )
            print(result)
            final_results.update(result)

        readable = remap_keys_to_labels(final_results, template)
        return readable




