import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Pinecone settings
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_region = os.getenv('PINECONE_REGION')
        self.pinecone_index = os.getenv('PINECONE_INDEX', 'code-embeddings')  # Added with default
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Repository settings
        self.repo_cache_dir = os.getenv('REPO_CACHE_DIR', 'cache/repos')
        self.repo_metadata_file = os.getenv('REPO_METADATA_FILE', 'cache/repo_metadata.json')
        
        # Chunking settings
        self.chunks_dir = os.getenv('CHUNKS_DIR', 'cache/chunks')
        self.chunks_metadata_file = os.getenv('CHUNKS_METADATA_FILE', 'cache/chunks_metadata.json')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        
        # File processing settings
        self.default_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.h', 
            '.cs', '.php', '.rb', '.go', '.txt', '.md',
            '.rst', '.json', '.yml', '.yaml', '.toml',
            '.ini', '.cfg', '.conf'
        }