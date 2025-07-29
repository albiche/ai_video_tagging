from abc import ABC, abstractmethod

class AudioExtractor(ABC):
    @abstractmethod
    def extract(self, video_path: str, output_dir: str) -> str:
        """
        Extract audio from a video and save it in output_dir.
        Return the saved audio file path.
        """
        pass
