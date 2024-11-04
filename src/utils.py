import mimetypes
import chardet
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        return cls(
            content=data["content"],
            metadata=data["metadata"]
        )

class FileTypeDetector:
    @staticmethod
    def is_text_file(file_path: str) -> bool:
        """Determine if a file is a text file using multiple methods."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and 'text' in mime_type:
            return True

        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                if not sample:
                    return True
                
                result = chardet.detect(sample)
                if result['encoding'] is not None:
                    sample.decode(result['encoding'])
                    return True
                
        except (UnicodeDecodeError, Exception):
            return False

        return False

    @staticmethod
    def get_encoding(file_path: str) -> str:
        """Detect the file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8' 