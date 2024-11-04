import time
import hashlib
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.config import Config
    from src.utils import Document, setup_logger
except ImportError:
    from config import Config
    from utils import Document, setup_logger

class PineconeUploader:
    def __init__(self, config: Config):
        """
        Initialize PineconeUploader with configuration.
        
        Args:
            config (Config): Configuration object containing Pinecone settings
        """
        self.logger = setup_logger('PineconeUploader')
        self.config = config
        self.index_name = config.pinecone_index
        self.embeddings = OpenAIEmbeddings()
        
        try:
            self.logger.info("Initializing Pinecone connection...")
            pc = PineconeGRPC(api_key=config.pinecone_api_key)
            
            # Create index if it doesn't exist
            if self.index_name not in pc.list_indexes().names():
                self.logger.info(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=config.pinecone_region
                    )
                )
            
            self.index = pc.Index(self.index_name)
            self.logger.info("Pinecone gRPC connection established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise

    def chunker(self, seq, size):
        """Utility function to create batches."""
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def upload_documents(self, documents: List[Document], namespace: str = "") -> None:
        """
        Upload documents to Pinecone index with optimized batch processing.
        
        Args:
            documents (List[Document]): List of documents to upload
            namespace (str, optional): Pinecone namespace. Defaults to ""
        """
        try:
            self.logger.info(f"Starting document upload to namespace: {namespace or 'default'}")
            total_docs = len(documents)
            self.logger.info(f"Processing {total_docs} documents")

            # Calculate optimal batch size based on vector dimension and metadata size
            # For 1536 dimensions with metadata, optimal batch size is around 200
            batch_size = 200

            # Process embeddings in parallel first
            self.logger.info("Generating embeddings in parallel...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_doc = {
                    executor.submit(self._create_vector, doc): doc 
                    for doc in documents
                }
                
                vectors = []
                with tqdm(total=total_docs, desc="Generating embeddings") as pbar:
                    for future in as_completed(future_to_doc):
                        try:
                            vector = future.result()
                            if vector:
                                vectors.append(vector)
                            pbar.update(1)
                        except Exception as e:
                            self.logger.error(f"Error processing document: {str(e)}")

            # Batch upload vectors using gRPC
            self.logger.info("Uploading vectors in parallel batches...")
            async_results = []
            with tqdm(total=len(vectors), desc="Uploading vectors") as pbar:
                for batch in self.chunker(vectors, batch_size):
                    try:
                        # Use async requests for parallel processing
                        result = self.index.upsert(
                            vectors=batch,
                            namespace=namespace,
                            async_req=True  # Enable async requests
                        )
                        async_results.append(result)
                        pbar.update(len(batch))
                    except Exception as e:
                        self.logger.error(f"Error uploading batch: {str(e)}")

            # Wait for and check all async results
            self.logger.info("Waiting for all uploads to complete...")
            for result in async_results:
                try:
                    result.result()  # Wait for and get the result
                except Exception as e:
                    self.logger.error(f"Async upload error: {str(e)}")

            self.logger.info("Document upload completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to upload documents: {str(e)}")
            raise

    def _create_vector(self, doc: Document) -> dict:
        """
        Create a vector from a document with retries.
        
        Args:
            doc (Document): Document to create vector from
            
        Returns:
            dict: Vector dictionary with id, values, and metadata
        """
        try:
            embedding = self.get_embedding(doc.content)
            return {
                'id': hashlib.md5(doc.content.encode()).hexdigest(),
                'values': embedding,
                'metadata': {
                    'content': doc.content,
                    **doc.metadata
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to create vector: {str(e)}")
            return None

    def get_embedding(self, text: str) -> List[float]:
        """
        Get OpenAI embedding for text with retry mechanism.
        
        Args:
            text (str): Text to get embedding for
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            Exception: If embedding fails after max retries
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embedding = self.embeddings.embed_query(text)
                return embedding
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                    raise
                self.logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(1)

    def delete_namespace(self, namespace: str) -> None:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace (str): Namespace to delete
            
        Raises:
            Exception: If deletion fails
        """
        try:
            self.logger.info(f"Deleting namespace: {namespace}")
            self.index.delete(delete_all=True, namespace=namespace)
            self.logger.info(f"Successfully deleted namespace: {namespace}")
        except Exception as e:
            self.logger.error(f"Failed to delete namespace: {str(e)}")
            raise