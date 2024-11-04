import os
import json
import hashlib
from typing import List, Dict
from .config import Config
from .utils import FileTypeDetector, Document
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, config: Config, model: str = "gpt-3.5-turbo"):
        self.config = config
        self.file_detector = FileTypeDetector()
        self.chunks_dir = config.chunks_dir
        self.cache_metadata_file = config.chunks_metadata_file
        
        # Create chunkers
        self.max_tokens = config.chunk_size
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=0,
            length_function=len,
            separators=["\nclass ", "\ndef ", "\n\n", "\n", ".", "?", "!", ";", ":", " ", ""]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=0,
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!", ";", ":", " ", ""]
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.encoding.encode(text))

    def get_file_hash(self, file_path: str) -> str:
        """Calculate hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def load_cache_metadata(self) -> Dict:
        """Load cache metadata if it exists."""
        if os.path.exists(self.cache_metadata_file):
            with open(self.cache_metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache_metadata(self, metadata: Dict) -> None:
        """Save cache metadata."""
        os.makedirs(os.path.dirname(self.cache_metadata_file), exist_ok=True)
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(metadata, f)

    def process_directory(self, directory: str) -> List[Document]:
        """Process all files in a directory and cache the chunks."""
        cache_metadata = self.load_cache_metadata()
        documents = []
        files_processed = False

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if self.should_process_file(file_path):
                    current_hash = self.get_file_hash(file_path)
                    cache_path = os.path.join(
                        self.chunks_dir, 
                        hashlib.md5(file_path.encode()).hexdigest() + ".json"
                    )

                    if (os.path.exists(cache_path) and 
                        file_path in cache_metadata and 
                        cache_metadata[file_path] == current_hash):
                        with open(cache_path, 'r') as f:
                            cached_chunks = json.load(f)
                            documents.extend([Document.from_dict(doc) for doc in cached_chunks])
                    else:
                        try:
                            chunks = self.process_file(file_path)
                            documents.extend(chunks)
                            
                            os.makedirs(self.chunks_dir, exist_ok=True)
                            with open(cache_path, 'w') as f:
                                json.dump([doc.to_dict() for doc in chunks], f)
                            
                            cache_metadata[file_path] = current_hash
                            files_processed = True
                        except Exception as e:
                            print(f"Error: {file_path} - {str(e)}")

        if files_processed:
            self.save_cache_metadata(cache_metadata)

        print(f"Success: Processed {len(documents)} documents")
        return documents

    def process_file(self, file_path: str) -> List[Document]:
        """Process a single file into chunks."""
        try:
            encoding = self.file_detector.get_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            splitter = self.code_splitter if self.is_code_file(file_path) else self.text_splitter
            chunks = splitter.split_text(content)

            return [
                Document(
                    content=chunk,
                    metadata={
                        "source": file_path,
                        "type": "code" if self.is_code_file(file_path) else "text",
                        "token_count": self.count_tokens(chunk)
                    }
                )
                for chunk in chunks
            ]
        except Exception as e:
            print(f"Error: {file_path} - {str(e)}")
            return []

    def should_process_file(self, file_path: str) -> bool:
        """Check if the file should be processed based on extension and content."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.default_extensions or self.file_detector.is_text_file(file_path)

    def is_code_file(self, file_path: str) -> bool:
        """Determine if the file is a code file based on extension."""
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go'}
        return os.path.splitext(file_path)[1].lower() in code_extensions

    def cache_chunks(self, documents: List[Document]) -> None:
        """Cache the processed chunks to disk."""
        os.makedirs(self.chunks_dir, exist_ok=True)
        cache_file = os.path.join(self.chunks_dir, "chunks.json")
        
        # Convert documents to serializable format
        serializable_docs = [doc.to_dict() for doc in documents]
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_docs, f)

    def load_cached_chunks(self) -> List[Document]:
        """Load cached chunks from disk."""
        cache_file = os.path.join(self.chunks_dir, "chunks.json")
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return [Document.from_dict(doc) for doc in data] 