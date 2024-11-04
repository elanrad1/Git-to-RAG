import os
from dotenv import load_dotenv

class Config:
    @classmethod
    def from_env(cls):
        """Class method to create Config instance from environment variables."""
        return cls()

    def __init__(self):
        load_dotenv()
        
        # Base paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = os.getenv('CACHE_DIR', os.path.join(self.base_dir, 'cache'))
        
        # Repository cache paths
        self.repo_cache_dir = os.path.join(self.cache_dir, "cloned_repo")
        self.repo_metadata_file = os.path.join(self.cache_dir, "repo_metadata.json")
        
        # Chunking settings
        self.chunks_dir = os.path.join(self.cache_dir, "chunks")
        self.chunks_metadata_file = os.path.join(self.chunks_dir, "metadata.json")
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.default_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.h', 
            '.cs', '.php', '.rb', '.go', '.txt', '.md',
            '.rst', '.json', '.yml', '.yaml', '.toml',
            '.ini', '.cfg', '.conf'
        }
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Pinecone settings
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
            
        self.pinecone_environment = os.getenv('PINECONE_ENV')
        self.pinecone_region = os.getenv('PINECONE_REGION', 'us-west-2')
        
        # Create necessary directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)