import os
import cv2
from ultralytics import YOLO
from frame_extractors.base_extractor import FrameExtractor


class PeopleExtractor(FrameExtractor):
    def __init__(self, interval_s=1.0, return_person_score=False):
        self.model = YOLO("yolov8n.pt")
        self.interval_s = interval_s
        self.return_person_score = return_person_score  # <--- AJOUT

    def detect_people_in_image(self, image_path):
        results = self.model(image_path)
        boxes = results[0].boxes
        person_boxes = [box for box in boxes if int(box.cls[0]) == 0]
        if not person_boxes:
            return False, [], 0.0

        image = cv2.imread(image_path)
        img_area = image.shape[0] * image.shape[1]

        total_person_area = 0
        bboxes = []

        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            total_person_area += area
            bboxes.append((x1, y1, x2, y2))

        person_area_ratio = total_person_area / img_area
        return True, bboxes, person_area_ratio

    def extract(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.interval_s)
        saved_frames = []
        count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                temp_path = os.path.join(output_dir, f"_tmp_frame.jpg")
                cv2.imwrite(temp_path, frame)
                has_person, _, person_area_ratio = self.detect_people_in_image(temp_path)
                if has_person:
                    final_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                    os.rename(temp_path, final_path)
                    saved_frames.append((final_path, person_area_ratio) if self.return_person_score else final_path)
                    frame_idx += 1
                else:
                    os.remove(temp_path)
            count += 1
        cap.release()
        return saved_frames
