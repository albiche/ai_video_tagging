import cv2
import os
from frame_extractors.base_extractor import FrameExtractor

class RegularExtractor(FrameExtractor):
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s

    def extract(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.interval_s)
        frame_idx = 0
        saved_frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
                frame_idx += 1
            count += 1
        cap.release()
        return saved_frames
