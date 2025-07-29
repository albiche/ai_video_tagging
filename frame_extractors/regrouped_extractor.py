import os
import cv2
import numpy as np
from typing import List
from frame_extractors.base_extractor import FrameExtractor

class RegroupedExtractor(FrameExtractor):
    def __init__(self, interval_s: float = 1.0, max_output_images: int = 10):
        """
        :param interval_s: Frame extraction interval in seconds
        :param max_output_images: Maximum number of grouped images to create
        """
        self.interval_s = interval_s
        self.max_output_images = max_output_images

    def _combine_frames(self, frames: List, frame_shape):
        """Combine multiple frames horizontally."""
        h, w = frame_shape[:2]
        combined_w = w * len(frames)
        combined = np.zeros((h, combined_w, 3), dtype=np.uint8)
        for idx, frame in enumerate(frames):
            combined[0:h, idx*w:(idx+1)*w] = frame
        return combined

    def extract(self, video_path: str, output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video '{video_path}'.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.interval_s)
        collected_frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                collected_frames.append(frame)

            count += 1

        cap.release()

        if not collected_frames:
            return []

        total_frames = len(collected_frames)
        frames_per_image = max(1, total_frames // self.max_output_images)
        frame_shape = collected_frames[0].shape

        grouped_images = []
        i = 0
        while i < total_frames:
            group = collected_frames[i:i+frames_per_image]
            if not group:
                break

            # If last group is smaller, fill with black frames
            while len(group) < frames_per_image:
                black_frame = np.zeros_like(collected_frames[0])
                group.append(black_frame)

            combined = self._combine_frames(group, frame_shape)
            grouped_images.append(combined)
            i += frames_per_image

        # Limit to max_output_images
        grouped_images = grouped_images[:self.max_output_images]

        saved_paths = []
        for idx, img in enumerate(grouped_images):
            path = os.path.join(output_dir, f"grouped_frame_{idx:04d}.jpg")
            cv2.imwrite(path, img)
            saved_paths.append(path)

        return saved_paths
