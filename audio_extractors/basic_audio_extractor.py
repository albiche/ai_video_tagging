import os
import moviepy as mp
from audio_extractors.base_extractor import AudioExtractor


class BasicAudioExtractor(AudioExtractor):
    def __init__(self, audio_format: str = "wav"):
        """
        :param audio_format: Output audio format ('wav', 'mp3', etc.)
        """
        self.audio_format = audio_format

    def extract(self, video_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)

        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio

        if audio_clip is None:
            raise ValueError(f"No audio track found in video '{video_path}'.")

        audio_filename = os.path.splitext(os.path.basename(video_path))[0] + f".{self.audio_format}"
        audio_path = os.path.join(output_dir, audio_filename)

        audio_clip.write_audiofile(audio_path, logger=None)  # no verbose

        # Clean up
        video_clip.close()
        audio_clip.close()

        return audio_path
