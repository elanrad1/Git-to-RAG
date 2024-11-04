from src.config import Config
from src.repo_cloner import RepoCloner
from src.chunker import Chunker
from src.pinecone_uploader import PineconeUploader

def main():
    # Repository settings
    repo_url = "https://github.com/IfcOpenShell/IfcOpenShell.git"
    target_folder = "src/ifcopenshell-python"  # Optional: specific folder to process
    model = "gpt-3.5-turbo"  # Model for tokenization
    index_name = "ifcopenshell-docs-v2"  # Pinecone index name

    # Initialize components
    config = Config.from_env()
    repo_cloner = RepoCloner(config, repo_url, target_folder)
    chunker = Chunker(config, model)
    uploader = PineconeUploader(config)

    # Process repository
    repo_path = repo_cloner.clone()
    target_path = repo_cloner.get_target_path()
    documents = chunker.process_directory(target_path)
    uploader.upload(documents, index_name)

if __name__ == "__main__":
    main() 