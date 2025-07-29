import os
import cv2
import numpy as np
from typing import List
from frame_extractors.base_extractor import FrameExtractor
from frame_extractors.face_extractor import PeopleExtractor

def compute_histogram_similarity(img1, img2) -> float:
    """Compute histogram similarity between two images (normalized correlation)."""
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist_img1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_img1, hist_img1)
    cv2.normalize(hist_img2, hist_img2)

    similarity = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    return similarity

class PeopleMIFExtractor(FrameExtractor):
    def __init__(self, max_frames: int = 10, interval_s: float = 0.5, similarity_threshold: float = 0.8):
        """
        :param max_frames: Maximum number of diverse people_1s frames to extract
        :param interval_s: Frame extraction interval in seconds
        :param similarity_threshold: Maximum allowed similarity between selected frames (1.0 = identical)
        """
        self.max_frames = max_frames
        self.interval_s = interval_s
        self.similarity_threshold = similarity_threshold

    def extract(self, video_path: str, output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Extract all frames with people_1s detection
        people_extractor = PeopleExtractor(interval_s=self.interval_s, return_person_score=True)
        frames_with_scores = people_extractor.extract(video_path, output_dir)

        if not frames_with_scores:
            return []

        # Step 2: Sort frames by people_1s area ratio (descending)
        frames_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Select frames ensuring visual diversity
        selected_frames = []
        selected_images = []

        for frame_path, _ in frames_with_scores:
            frame = cv2.imread(frame_path)

            # Check if frame is sufficiently different from already selected ones
            is_similar = False
            for selected_img in selected_images:
                similarity = compute_histogram_similarity(frame, selected_img)
                if similarity >= self.similarity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                selected_frames.append(frame_path)
                selected_images.append(frame)

            if len(selected_frames) >= self.max_frames:
                break

        # Step 4: Cleanup: remove unused frames from disk
        all_frame_paths = [fp for fp, _ in frames_with_scores]
        unused_frames = set(all_frame_paths) - set(selected_frames)

        for frame_path in unused_frames:
            if os.path.exists(frame_path):
                os.remove(frame_path)

        return selected_frames
