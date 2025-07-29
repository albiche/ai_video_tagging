from typing import List, Dict, Tuple
import json
import tiktoken


def estimate_tokens_from_messages(messages: List[Dict], model: str = "gpt-4") -> int:

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    total += len(enc.encode(part["text"]))
                elif part["type"] == "image_url":
                    total += 100  # OpenAI estimate: 85–120 tokens per image
    return total


def build_prompt_messages(
    fields_dict: Dict,
    images_b64: List[str],
    transcriptions: List[str] = None
) -> List[Dict]:

    fields = {
        k: {
            "description": v["prompt_ai"],
            "accepted_values": v.get("accepted_values", [])
        }
        for k, v in fields_dict.items()
    }

    # Prompt system avec détection de sources
    sources = []
    if images_b64:
        sources.append("frames")
    if transcriptions:
        sources.append("transcription")

    if not sources:
        raise ValueError("At least one of images or transcriptions must be provided.")

    system_prompt = (
        "You are an expert in marketing analysis for alcoholic beverage products.\n"
        f"You are given {', and '.join(sources)} from an advertisement.\n"
    )

    if transcriptions:
        system_prompt += "Transcription:\n" + "\n".join(transcriptions[:1]) + "\n\n"
        print("transcriptions",transcriptions)

    system_prompt += (
        "Your task is to extract structured information based on the provided material.\n"
        "Return a valid JSON dictionary with key: value pairs.\n"
        "Use only the keys and descriptions provided below. If a value is not identifiable, return 'N/A'.\n"
        "Respond only with the JSON object: {key: value, ...}.\n\n"
        f"Fields:\n{json.dumps(fields)}"
    )

    user_content = [{"type": "text", "text": f"Here {' and '.join(sources)}:"}]
    if images_b64:
        user_content += [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in images_b64
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]




def smart_split_prompt(
    prompt_data: Dict,
    images_b64: List[str],
    transcriptions: List[str] = None,
    max_tokens: int = 8000,
    model: str = "gpt-4",
    max_images_per_chunk: int = 6,
    max_chunks: int = 10,
    split_image: bool = True
) -> List[Tuple[Dict, List[str], List[str]]]:

    all_chunks = []
    transcript = transcriptions if transcriptions else []

    # Cas 1 : pas de split d’images
    if not split_image:
        images_b64 = images_b64[:max_images_per_chunk] if images_b64 else []
        current_fields = {}

        for key, val in prompt_data.items():
            test_fields = {**current_fields, key: val}
            token_estimate = estimate_tokens_from_messages(
                build_prompt_messages(test_fields, images_b64, transcript),
                model
            )
            print(f"Estimated tokens for {list(test_fields.keys())}: {token_estimate}")

            if token_estimate > max_tokens:
                if not current_fields:
                    all_chunks.append(({key: val}, images_b64, transcript))
                else:
                    all_chunks.append((current_fields.copy(), images_b64, transcript))
                    current_fields = {key: val}
            else:
                current_fields[key] = val

        if current_fields:
            all_chunks.append((current_fields.copy(), images_b64, transcript))
        return all_chunks

    # Cas ou y a que l'audio
    if not images_b64:
        current_fields = {}
        for key, val in prompt_data.items():
            test_fields = {**current_fields, key: val}
            token_estimate = estimate_tokens_from_messages(
                build_prompt_messages(test_fields, [], transcript),
                model
            )
            print(f"Estimated tokens (audio only) for {list(test_fields.keys())}: {token_estimate}")

            if token_estimate > max_tokens:
                if not current_fields:
                    all_chunks.append(({key: val}, [], transcript))
                else:
                    all_chunks.append((current_fields.copy(), [], transcript))
                    current_fields = {key: val}
            else:
                current_fields[key] = val

        if current_fields:
            all_chunks.append((current_fields.copy(), [], transcript))
        return all_chunks

    # Cas 2 : split des images, transcription inchangée
    total_images = len(images_b64) if images_b64 else 0
    for i in range(0, total_images, max_images_per_chunk):
        image_chunk = images_b64[i:i + max_images_per_chunk]
        current_fields = {}
        field_chunks = []

        for key, val in prompt_data.items():
            test_fields = {**current_fields, key: val}
            token_estimate = estimate_tokens_from_messages(
                build_prompt_messages(test_fields, image_chunk, transcript),
                model
            )
            if token_estimate > max_tokens:
                if not current_fields:
                    field_chunks.append(({key: val}, image_chunk, transcript))
                else:
                    field_chunks.append((current_fields.copy(), image_chunk, transcript))
                    current_fields = {key: val}
            else:
                current_fields[key] = val

        if current_fields:
            field_chunks.append((current_fields.copy(), image_chunk, transcript))

        all_chunks.extend(field_chunks)

    if len(all_chunks) > max_chunks:
        print(f"❌ Skipping prompt: {len(all_chunks)} chunks needed (max allowed is {max_chunks}).")
        return []

    return all_chunks


def merge_responses(
    responses: List[Dict],
    batch_config: Dict,
    frames_per_chunk: List[int] = None,
    ratios: Dict[str, float] = None,
    current_frame_method: str = None
) -> Dict:
    final = {}
    collected: Dict[str, List[str]] = {}
    weights = frames_per_chunk if frames_per_chunk else [1] * len(responses)
    ratios = ratios or {}

    for idx, r in enumerate(responses):
        for k, v in r.items():
            if k not in collected:
                collected[k] = []
            collected[k].append((str(v), weights[idx]))

    for k, pairs in collected.items():
        values, chunk_weights = zip(*pairs)
        logic = batch_config.get(k, {}).get("split_logic", "or")

        if logic == "or":
            final[k] = "1" if "1" in values else "0"

        elif logic == "and":
            final[k] = "1" if all(v == "1" for v in values) else "0"

        elif logic == "add":
            try:
                total = sum(int(v) for v in values if v.isdigit())
                final[k] = str(min(total, 100))
            except Exception:
                final[k] = "N/A"

        elif logic == "mean":
            try:
                total_weighted = sum(int(v) * w for v, w in zip(values, chunk_weights) if v.isdigit())
                total_weight = sum(w for v, w in zip(values, chunk_weights) if v.isdigit())
                avg = total_weighted / total_weight if total_weight > 0 else 0
                final[k] = str(int(round(avg)))
            except Exception:
                final[k] = "N/A"

        elif logic == "count-mean":
            try:
                total_count = sum(int(v) for v in values if v.isdigit())
                total_frames = sum(chunk_weights)
                percent = (100 * total_count / total_frames) if total_frames > 0 else 0
                final[k] = str(int(round(percent)))
            except Exception:
                final[k] = "N/A"

        elif logic == "mean-total":
            try:
                ratio = ratios.get(current_frame_method, 1.0)
                total_weighted = sum(int(v) * w for v, w in zip(values, chunk_weights) if v.isdigit())
                total_weight = sum(w for v, w in zip(values, chunk_weights) if v.isdigit())
                avg = (total_weighted / total_weight) * ratio if total_weight > 0 else 0
                final[k] = str(int(round(avg)))
            except Exception:
                final[k] = "N/A"

        elif logic == "count-mean-total":
            try:
                ratio = ratios.get(current_frame_method, 1.0)
                total_count = sum(int(v) for v in values if v.isdigit())
                total_frames = sum(chunk_weights)
                percent = (100 * total_count / total_frames) * ratio if total_frames > 0 else 0
                final[k] = str(int(round(percent)))
            except Exception:
                final[k] = "N/A"

        else:
            final[k] = values[0]

    return final


