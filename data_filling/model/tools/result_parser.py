def build_prompt_json(batch_config: dict) -> dict:
    prompt = {}
    for key, info in batch_config.items():
        prompt[key] = {
            "description": info["prompt_ai"],
            "accepted_values": info["accepted_values"]
        }
    return prompt
def parse_gpt_output(response: str, batch_config: dict) -> dict:
    """
    Parses the GPT response string into a clean dictionary of results.
    """
    try:
        parsed = eval(response) if isinstance(response, str) else response
        return {key: parsed.get(key, "N/A") for key in batch_config.keys()}
    except Exception:
        return {key: "N/A" for key in batch_config.keys()}
