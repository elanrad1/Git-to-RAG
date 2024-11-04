from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import hashlib
import time

from .config import Config
from .utils import setup_logger
from langchain.schema import Document

class PineconeUploader:
    def __init__(self, config: Config):
        self.logger = setup_logger('PineconeUploader')
        self.config = config
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone
        pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = pc.Index(config.pinecone_index)
        
        # Batch settings
        self.batch_size = 50  # Adjust based on your rate limits

    def upload_documents(self, documents: List[Document], namespace: str = "") -> None:
        """Upload documents with parallel processing."""
        try:
            self.logger.info(f"Processing {len(documents)} documents")
            
            # Step 1: Generate embeddings in parallel
            vectors = []
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(self._create_vector, doc) for doc in documents]
                
                for future in tqdm(futures, desc="Generating embeddings"):
                    vector = future.result()
                    if vector:
                        vectors.append(vector)

            # Step 2: Batch upload to Pinecone
            total_vectors = len(vectors)
            for i in tqdm(range(0, total_vectors, self.batch_size), desc="Uploading to Pinecone"):
                batch = vectors[i:i + self.batch_size]
                self._upload_batch(batch, namespace)

            self.logger.info("Upload completed successfully")

        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            raise

    def _create_vector(self, doc: Document) -> dict:
        """Create a vector with retries."""
        for attempt in range(3):  # 3 retries
            try:
                embedding = self.embeddings.embed_query(doc.content)
                return {
                    'id': hashlib.md5(doc.content.encode()).hexdigest(),
                    'values': embedding,
                    'metadata': {
                        'content': doc.content,
                        **doc.metadata
                    }
                }
            except Exception as e:
                if attempt == 2:  # Last attempt
                    self.logger.error(f"Failed to create vector: {str(e)}")
                    return None
                time.sleep(1)  # Wait before retry

    def _upload_batch(self, batch: List[dict], namespace: str) -> None:
        """Upload a batch with retries."""
        for attempt in range(3):  # 3 retries
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                return
            except Exception as e:
                if attempt == 2:  # Last attempt
                    self.logger.error(f"Failed to upload batch: {str(e)}")
                    raise
                time.sleep(1)  # Wait before retry