import os
import json
from git import Repo, GitCommandError
import shutil
from typing import Dict
import time
import hashlib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.config import Config
    from src.utils import setup_logger
except ImportError:
    from config import Config
    from utils import setup_logger

class RepoCloner:
    def __init__(self, config: Config, repo_url: str, target_folder: str = ""):
        self.logger = setup_logger('RepoCloner')
        self.config = config
        self.repo_url = repo_url
        self.target_folder = target_folder.replace('\\', '/').strip('/')  # Normalize path
        
        # Create a unique directory name based on repo URL and target folder
        repo_hash = hashlib.md5(f"{repo_url}:{target_folder}".encode()).hexdigest()[:8]
        self.clone_dir = os.path.join(config.repo_cache_dir, repo_hash)
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
        if not os.path.exists(directory):
            return True
            
        for i in range(max_retries):
            try:
                # First try to remove read-only flags if on Windows
                if os.name == 'nt':  # Windows
                    for root, dirs, files in os.walk(directory):
                        for dir_path in dirs:
                            os.chmod(os.path.join(root, dir_path), 0o777)
                        for file_path in files:
                            os.chmod(os.path.join(root, file_path), 0o777)
                
                shutil.rmtree(directory)
                
                # Verify directory is actually removed
                if not os.path.exists(directory):
                    return True
                    
                time.sleep(1)  # Wait before retry
            except Exception as e:
                self.logger.error(f"Attempt {i+1} failed to remove directory: {str(e)}")
                if i == max_retries - 1:  # Last attempt
                    self.logger.error(f"Error: Could not remove directory after {max_retries} attempts")
                    return False
                time.sleep(1)  # Wait before retry
        return False

    def get_target_path(self) -> str:
        """Get the full path to the target folder."""
        if self.target_folder:
            target_path = os.path.join(self.clone_dir, self.target_folder)
            # Ensure the target path exists and is a directory
            if not os.path.exists(target_path):
                self.logger.error(f"Target folder not found: {self.target_folder}")
                raise ValueError(f"Target folder not found: {self.target_folder}")
            if not os.path.isdir(target_path):
                self.logger.error(f"Target path is not a directory: {self.target_folder}")
                raise ValueError(f"Target path is not a directory: {self.target_folder}")
            self.logger.info(f"Using target folder: {target_path}")
            return target_path
        return self.clone_dir

    def check_target_folder(self) -> bool:
        """Check if target folder exists and is valid."""
        if not self.target_folder:
            return True
        target_path = os.path.join(self.clone_dir, self.target_folder)
        exists = os.path.exists(target_path)
        is_dir = os.path.isdir(target_path)
        if not exists:
            self.logger.error(f"Target folder does not exist: {self.target_folder}")
        elif not is_dir:
            self.logger.error(f"Target path is not a directory: {self.target_folder}")
        return exists and is_dir

    def clone(self) -> str:
        """Clone the repository and return the target folder path."""
        metadata = self.load_metadata()
        current_url = metadata.get('repo_url', '')
        current_hash = metadata.get('commit_hash', '')

        self.logger.debug(f"Cache check - Current URL: {current_url}")
        self.logger.debug(f"Cache check - Current Hash: {current_hash}")
        self.logger.debug(f"Cache check - Directory exists: {os.path.exists(self.clone_dir)}")

        if (os.path.exists(self.clone_dir) and 
            current_url == self.repo_url and 
            current_hash == self.get_repo_hash()):
            self.logger.info(f"Using cached repository: {self.repo_url}")
            
            if self.target_folder and not self.check_target_folder():
                self.logger.warning(f"Target folder '{self.target_folder}' not found in cached repo. Re-cloning...")
                self.safe_remove_directory(self.clone_dir)
            else:
                return self.get_target_path()

        if not self.safe_remove_directory(self.clone_dir):
            self.logger.error("Could not clear existing repository directory")
            raise Exception("Could not clear existing repository directory")

        try:
            self.logger.info(f"Cloning repository: {self.repo_url}")
            os.makedirs(os.path.dirname(self.clone_dir), exist_ok=True)
            
            try:
                self.logger.info("Attempting HTTPS clone...")
                repo = Repo.clone_from(
                    self.repo_url, 
                    self.clone_dir,
                    depth=1
                )
            except GitCommandError as e:
                ssh_url = self.repo_url.replace('https://github.com/', 'git@github.com:')
                self.logger.warning(f"HTTPS clone failed, trying SSH: {ssh_url}")
                repo = Repo.clone_from(
                    ssh_url, 
                    self.clone_dir,
                    depth=1
                )
            
            if self.target_folder and not self.check_target_folder():
                self.logger.error(f"Target folder '{self.target_folder}' not found in cloned repository")
                raise Exception(f"Target folder '{self.target_folder}' not found in cloned repository")
            
            metadata = {
                'repo_url': self.repo_url,
                'commit_hash': repo.head.commit.hexsha,
                'target_folder': self.target_folder,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_metadata(metadata)
            
            target_path = self.get_target_path()
            self.logger.info(f"Success: Repository cloned to {self.clone_dir}")
            self.logger.info(f"Using target path: {target_path}")
            return target_path
            
        except GitCommandError as e:
            self.logger.error(f"Git clone failed: {str(e)}")
            self.logger.error("\nPlease ensure:")
            self.logger.error("1. The repository URL is correct")
            self.logger.error("2. You have access to the repository")
            self.logger.error("3. Git is installed and configured properly")
            self.logger.error("4. Your internet connection is stable")
            self.safe_remove_directory(self.clone_dir)
            raise
        except Exception as e:
            self.logger.error(f"Failed to clone repository: {str(e)}")
            self.safe_remove_directory(self.clone_dir)
            raise