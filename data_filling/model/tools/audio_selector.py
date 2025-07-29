def select_audio(audio_paths: list) -> str | None:
    """
    Selects an audio file from the list.
    For now, just take the first one. You can customize this if needed.
    """
    return audio_paths[0] if audio_paths else None
