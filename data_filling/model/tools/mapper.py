from typing import List, Dict, Tuple


def remap_keys_to_labels(model_output: Dict[str, str], template: Dict[str, dict]) -> Dict[str, str]:
    readable_output = {}
    for label, meta in template.items():
        key = meta.get("key")
        readable_output[label] = model_output.get(key, "N/A")
    return readable_output
