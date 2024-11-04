import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.repo_cloner import RepoCloner
from src.chunker import Chunker
from src.pinecone_uploader import PineconeUploader

def main():
    # Initialize configuration
    config = Config()
    
    # Repository settings
    repo_url = "https://github.com/IfcOpenShell/IfcOpenShell.git"
    target_folder = "src/ifcopenshell-python"
    index_name = "ifcopenshell-python"
    
    try:
        # Set the index name in config
        config.pinecone_index = index_name
        
        # Initialize components
        cloner = RepoCloner(config, repo_url, target_folder)
        chunker = Chunker(config)
        uploader = PineconeUploader(config)
        
        # Process repository
        repo_path = cloner.clone()
        documents = chunker.process_directory(repo_path)
        uploader.upload_documents(documents, namespace="my-repo")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 