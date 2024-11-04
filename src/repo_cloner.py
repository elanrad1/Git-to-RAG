import os
import json
from git import Repo
from .config import Config
import shutil
from typing import Dict
import time

class RepoCloner:
    def __init__(self, config: Config, repo_url: str, target_folder: str = ""):
        self.config = config
        self.repo_url = repo_url
        self.target_folder = target_folder
        self.clone_dir = config.repo_cache_dir
        self.metadata_file = config.repo_metadata_file

    def get_repo_hash(self) -> str:
        """Get a hash representing the current state of the repository."""
        try:
            repo = Repo(self.clone_dir)
            return repo.head.commit.hexsha
        except:
            return ""

    def load_metadata(self) -> dict:
        """Load repository metadata if it exists."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self, metadata: Dict) -> None:
        """Save repository metadata."""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)

    def safe_remove_directory(self, directory: str, max_retries: int = 3) -> bool:
        """Safely remove a directory with retries."""
        for i in range(max_retries):
            try:
                if os.path.exists(directory):
                    shutil.rmtree(directory, ignore_errors=True)
                return True
            except Exception as e:
                if i == max_retries - 1:  # Last attempt
                    print(f"Error: Could not remove directory after {max_retries} attempts - {str(e)}")
                    return False
                time.sleep(1)  # Wait before retry
        return False

    def clone(self) -> str:
        """
        Clone the repository if it doesn't exist in cache or has changed.
        Returns the path to the cloned repository.
        """
        metadata = self.load_metadata()
        current_url = metadata.get('repo_url', '')
        current_hash = metadata.get('commit_hash', '')

        # Check if repo exists and is the same
        if (os.path.exists(self.clone_dir) and 
            current_url == self.repo_url and 
            current_hash == self.get_repo_hash()):
            return self.clone_dir

        # Remove existing repo if it exists
        if not self.safe_remove_directory(self.clone_dir):
            raise Exception("Could not clear existing repository directory")

        try:
            os.makedirs(os.path.dirname(self.clone_dir), exist_ok=True)
            repo = Repo.clone_from(self.repo_url, self.clone_dir)
            
            metadata = {
                'repo_url': self.repo_url,
                'commit_hash': repo.head.commit.hexsha
            }
            self.save_metadata(metadata)
            
            print("Success: Repository cloned")
            return self.clone_dir
            
        except Exception as e:
            print(f"Error: Failed to clone repository - {str(e)}")
            self.safe_remove_directory(self.clone_dir)
            raise

    def get_target_path(self) -> str:
        """Get the path to the target folder within the repository."""
        return os.path.join(self.clone_dir, self.target_folder)