from frame_extractors.regular_extractor import RegularExtractor
from frame_extractors.mif_extractor import MIFExtractor
from frame_extractors.face_extractor import PeopleExtractor
from frame_extractors.regrouped_extractor import RegroupedExtractor
from frame_extractors.people_mif_extractor import PeopleMIFExtractor
from data_filling.pipeline.tools_pipeline.utils import ensure_dir
from audio_extractors.basic_audio_extractor import BasicAudioExtractor
import os




def get_video_id(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]

def extract_all_framings(video_path: str, output_dir: str) -> tuple:
    video_id = get_video_id(video_path)
    video_output_dir = os.path.join(output_dir, "extracted_frames", video_id)

    if os.path.exists(video_output_dir):
        print(f"📁 Using cached frames for video: {video_id}")
        paths = {
            method: [os.path.join(method_dir, f) for f in sorted(os.listdir(method_dir))]
            for method in os.listdir(video_output_dir)
            if os.path.isdir(method_dir := os.path.join(video_output_dir, method))
        }
    else:
        print(f"🧪 Extracting frames and audio for video: {video_id}")
        ensure_dir(video_output_dir)

        paths = {
            "regular_1s": RegularExtractor(interval_s=1.0).extract(
                video_path, os.path.join(video_output_dir, "regular_1s")
            ),
            "regular_0_5s": RegularExtractor(interval_s=0.5).extract(
                video_path, os.path.join(video_output_dir, "regular_0_5s")
            ),
            "mif": MIFExtractor(max_frames=10).extract(
                video_path, os.path.join(video_output_dir, "mif")
            ),
            "people_1s": PeopleExtractor(interval_s=1.0).extract(
                video_path, os.path.join(video_output_dir, "people_1s")
            ),
            "people_0_5s": PeopleExtractor(interval_s=0.5).extract(
                video_path, os.path.join(video_output_dir, "people_0_5s")
            ),
            "people_mif": PeopleMIFExtractor(max_frames=10, interval_s=0.5).extract(
                video_path, os.path.join(video_output_dir, "people_mif")
            ),
            "regroup_1s": RegroupedExtractor(interval_s=1.0, max_output_images=10).extract(
                video_path, os.path.join(video_output_dir, "regroup_1s")
            ),
        }

        # Audio extraction
        audio_path = BasicAudioExtractor(audio_format="wav").extract(
            video_path, os.path.join(video_output_dir, "audio")
        )
        paths["audio"] = [audio_path]

    return video_id, paths
