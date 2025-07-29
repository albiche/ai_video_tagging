from typing import List
import random

def select_frames(frames: List[str], frames_used: str) -> List[str]:
    if not frames:
        return []

    if frames_used == "all":
        return frames
    elif frames_used == "6_first":
        return frames[:6]
    elif frames_used == "5_last":
        return frames[-5:]
    elif frames_used == "random_10":
        return random.sample(frames, min(10, len(frames)))
    else:
        raise ValueError(f"Unsupported frames_used value: {frames_used}")


def select_audio( audio_paths: list) -> str | None:
    """
    Selects an audio file from the list.
    For now, just take the first one. You can customize this if needed.
    """
    return audio_paths[0] if audio_paths else None
