import os
import cv2
import numpy as np
import statistics
from typing import List
from frame_extractors.base_extractor import FrameExtractor


def is_uniform(frame, threshold_std: float = 5.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.std(gray) < threshold_std


import os
import cv2
import numpy as np
import statistics
from typing import List
from frame_extractors.base_extractor import FrameExtractor


def is_uniform(frame, threshold_std: float = 5.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.std(gray) < threshold_std


class MIFExtractor(FrameExtractor):
    def __init__(self, max_frames: int = 10, k: float = 4.0):
        """
        :param max_frames: maximum number of frames to extract
        :param k: multiplier for selecting frames with high difference
        """
        self.max_frames = max_frames
        self.k = k

    def extract(self, video_path: str, output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video '{video_path}'.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        nb_frames_target = min(self.max_frames, max(1, int(duration_sec)))
        min_frames = min(3, nb_frames_target)

        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error: Empty or corrupted video '{video_path}'.")
            return []

        frames = [prev_frame]
        diffs = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            prev_gray = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_val = np.sum(cv2.absdiff(prev_gray, gray))
            diffs.append((diff_val, len(frames) - 1))

        cap.release()

        if not diffs:
            return []

        diff_values = [float(d[0]) for d in diffs]
        threshold = statistics.mean(diff_values) + self.k * statistics.pstdev(diff_values)
        selected = [idx for (val, idx) in diffs if val >= threshold]

        if 0 not in selected:
            selected.insert(0, 0)
        if len(frames) - 1 not in selected:
            selected.append(len(frames) - 1)

        selected = sorted(set(selected))

        if len(selected) < min_frames:
            top_diffs = sorted(diffs, key=lambda x: x[0], reverse=True)
            needed = min_frames - len(selected)
            for val, idx in top_diffs:
                if idx not in selected:
                    selected.append(idx)
                    needed -= 1
                    if needed == 0:
                        break
            selected = sorted(set(selected))

        if len(selected) > nb_frames_target:
            diff_dict = {idx: val for (val, idx) in diffs if idx in selected}
            sorted_by_diff = sorted(selected, key=lambda x: diff_dict.get(x, 0), reverse=True)
            selected = sorted(set(sorted_by_diff[:nb_frames_target]))

        saved_paths = []
        for i, idx in enumerate(selected):
            frame = frames[idx]
            if is_uniform(frame):
                continue
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)

        return saved_paths

