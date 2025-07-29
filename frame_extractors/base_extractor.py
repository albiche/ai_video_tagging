from abc import ABC, abstractmethod

class FrameExtractor(ABC):
    @abstractmethod
    def extract(self, video_path: str, output_dir: str) -> list:
        """
        Extract frames from a video and save them in output_dir.
        Return list of saved frame paths.
        """
        pass
