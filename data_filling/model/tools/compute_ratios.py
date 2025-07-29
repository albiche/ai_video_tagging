def compute_frame_ratios(frames_by_method: dict) -> dict:
    """
    Args:
        frames_by_method (dict): {"regular_1s": [...], "people_0_5s": [...], "regrouped_1s": [...], ...}

    Returns:
        dict: {
            "regular_1s_total": int,
            "ratio_people_1s": float,
            "ratio_people_0_5s": float,
            "ratio_regrouped_1s": float,
            ...
        }
    """
    ratios = {}
    regular_1s_frames = len(frames_by_method.get("regular_1s", []))
    if regular_1s_frames == 0:
        print("⚠️ No regular_1s frames found, defaulting ratios to 1.")
        regular_1s_frames = 1  # Avoid division by 0, fallback

    ratios["regular_1s_total"] = regular_1s_frames

    for method in frames_by_method:
        total = len(frames_by_method[method])
        ratios[f"ratio_{method}"] = total / regular_1s_frames

    return ratios
