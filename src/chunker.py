import os
import json
import hashlib
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.config import Config
    from src.utils import FileTypeDetector, Document, setup_logger
except ImportError:
    from config import Config
    from utils import FileTypeDetector, Document, setup_logger

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, config: Config, model: str = "gpt-3.5-turbo", repo_url: str = None):
        self.logger = setup_logger('Chunker')
        self.config = config
        self.file_detector = FileTypeDetector()
        self.chunks_dir = config.chunks_dir
        self.cache_metadata_file = config.chunks_metadata_file
        self.repo_url = repo_url
        
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

    def get_repo_specific_chunks_dir(self) -> str:
        """Get a repository-specific cache directory."""
        if self.repo_url:
            repo_hash = hashlib.md5(self.repo_url.encode()).hexdigest()[:8]
            return os.path.join(self.chunks_dir, repo_hash)
        return self.chunks_dir

    def process_directory(self, directory: str) -> List[Document]:
        """Process all files in a directory and cache the chunks."""
        if not os.path.exists(directory):
            self.logger.error(f"Directory does not exist: {directory}")
            raise ValueError(f"Directory does not exist: {directory}")
            
        self.logger.info(f"Processing directory: {directory}")
        chunks_dir = self.get_repo_specific_chunks_dir()
        os.makedirs(chunks_dir, exist_ok=True)
        
        cache_metadata = self.load_cache_metadata()
        documents = []
        files_processed = False

        # Use the target directory as the base for relative paths
        base_dir = directory

        # Process only files within the target directory
        for root, _, files in os.walk(directory):
            # Skip if this is a hidden directory or inside .git
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
                
            for file in files:
                # Skip hidden files
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                if self.should_process_file(file_path):
                    # Create relative path from the target directory
                    relative_path = os.path.relpath(file_path, base_dir)
                    
                    current_hash = self.get_file_hash(file_path)
                    cache_key = f"{self.repo_url}:{relative_path}" if self.repo_url else relative_path
                    cache_path = os.path.join(
                        chunks_dir,
                        hashlib.md5(cache_key.encode()).hexdigest() + ".json"
                    )

                    if (os.path.exists(cache_path) and 
                        cache_key in cache_metadata and 
                        cache_metadata[cache_key] == current_hash):
                        self.logger.debug(f"Using cached chunks for: {relative_path}")
                        with open(cache_path, 'r') as f:
                            cached_chunks = json.load(f)
                            documents.extend([Document.from_dict(doc) for doc in cached_chunks])
                    else:
                        try:
                            self.logger.info(f"Processing file: {relative_path}")
                            chunks = self.process_file(file_path, relative_path)
                            documents.extend(chunks)
                            
                            with open(cache_path, 'w') as f:
                                json.dump([doc.to_dict() for doc in chunks], f)
                            
                            cache_metadata[cache_key] = current_hash
                            files_processed = True
                        except Exception as e:
                            self.logger.error(f"Error processing {relative_path}: {str(e)}")

        if files_processed:
            self.save_cache_metadata(cache_metadata)

        self.logger.info(f"Successfully processed {len(documents)} documents from {directory}")
        return documents

    def process_file(self, file_path: str, relative_path: str) -> List[Document]:
        """Process a single file into chunks."""
        try:
            encoding = self.file_detector.get_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Choose appropriate splitter based on file type
            if self.is_code_file(file_path):
                splitter = self.code_splitter
                doc_type = "code"
            else:
                splitter = self.text_splitter
                doc_type = "text"

            chunks = splitter.split_text(content)

            return [
                Document(
                    content=chunk,
                    metadata={
                        "source": relative_path,
                        "type": doc_type,
                        "token_count": self.count_tokens(chunk)
                    }
                )
                for chunk in chunks
            ]
        except Exception as e:
            self.logger.error(f"Error processing {relative_path}: {str(e)}")
            return []

    def should_process_file(self, file_path: str) -> bool:
        """Check if the file should be processed based on extension and content."""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.config.default_extensions or self.file_detector.is_text_file(file_path)

    def is_code_file(self, file_path: str) -> bool:
        """Determine if the file is a code file based on extension."""
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go'}
        return os.path.splitext(file_path)[1].lower() in code_extensions

    def is_rst_file(self, file_path: str) -> bool:
        """Determine if the file is an RST file."""
        return os.path.splitext(file_path)[1].lower() == '.rst'

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