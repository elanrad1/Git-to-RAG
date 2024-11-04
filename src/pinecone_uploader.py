from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import hashlib
import time
import queue
import threading

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
        
        # Queues and settings
        self.vector_queue = queue.Queue(maxsize=100)
        self.upload_queue = queue.Queue(maxsize=20)  # Queue for batch uploads
        self.batch_size = 50
        self.upload_workers = 3  # Number of parallel upload workers
        
        # Progress tracking
        self.total_processed = 0
        self.total_uploaded = 0
        self.upload_lock = threading.Lock()

    def upload_documents(self, documents: List[Document], namespace: str = "") -> None:
        """Process and upload documents with parallel processing."""
        try:
            total_docs = len(documents)
            self.logger.info(f"Processing {total_docs} documents")

            # Start multiple upload workers
            upload_threads = []
            for _ in range(self.upload_workers):
                thread = threading.Thread(
                    target=self._upload_worker,
                    args=(namespace, total_docs)
                )
                thread.start()
                upload_threads.append(thread)

            # Start batch collector thread
            batch_thread = threading.Thread(
                target=self._batch_collector,
                args=(total_docs,)
            )
            batch_thread.start()

            # Process embeddings in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                list(tqdm(
                    executor.map(self._process_and_queue_document, documents),
                    total=total_docs,
                    desc="Generating embeddings"
                ))

            # Signal batch collector to finish
            self.vector_queue.put(None)
            batch_thread.join()

            # Signal upload workers to finish
            for _ in range(self.upload_workers):
                self.upload_queue.put(None)
            for thread in upload_threads:
                thread.join()

            self.logger.info("Processing and upload completed successfully")

        except Exception as e:
            self.logger.error(f"Process failed: {str(e)}")
            raise

    def _process_and_queue_document(self, doc: Document) -> None:
        """Process a single document and add to queue."""
        try:
            vector = self._create_vector(doc)
            if vector:
                self.vector_queue.put(vector)
                self.total_processed += 1
        except Exception as e:
            self.logger.error(f"Failed to process document: {str(e)}")

    def _batch_collector(self, total_docs: int) -> None:
        """Collects vectors into batches for parallel upload."""
        batch = []
        while True:
            try:
                vector = self.vector_queue.get(timeout=60)
                
                if vector is None:  # End signal
                    if batch:  # Queue final batch
                        self.upload_queue.put(batch)
                    break

                batch.append(vector)
                
                # Queue when batch is full
                if len(batch) >= self.batch_size:
                    self.upload_queue.put(batch)
                    batch = []

            except queue.Empty:
                if batch:  # Queue partial batch after timeout
                    self.upload_queue.put(batch)
                    batch = []
            except Exception as e:
                self.logger.error(f"Batch collector error: {str(e)}")

    def _upload_worker(self, namespace: str, total_docs: int) -> None:
        """Worker thread to handle uploads in parallel."""
        pbar = tqdm(total=total_docs, desc="Uploading to Pinecone")
        last_update = 0

        while True:
            try:
                batch = self.upload_queue.get(timeout=60)
                
                if batch is None:  # End signal
                    break

                self._upload_batch(batch, namespace)
                
                # Update progress bar
                with self.upload_lock:
                    current = self.total_uploaded
                    pbar.update(current - last_update)
                    last_update = current

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Upload worker error: {str(e)}")
                
        pbar.close()

    def _create_vector(self, doc: Document) -> dict:
        """Create a vector with retries."""
        for attempt in range(3):
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
                if attempt == 2:
                    self.logger.error(f"Failed to create vector: {str(e)}")
                    return None
                time.sleep(1)

    def _upload_batch(self, batch: List[dict], namespace: str) -> None:
        """Upload a batch with retries."""
        for attempt in range(3):
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                with self.upload_lock:
                    self.total_uploaded += len(batch)
                return
            except Exception as e:
                if attempt == 2:
                    self.logger.error(f"Failed to upload batch: {str(e)}")
                    raise
                time.sleep(1)