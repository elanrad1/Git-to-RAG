from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC
from typing import List
from .config import Config
from .utils import Document
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class PineconeUploader:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.openai_api_key
        )
        
        # Initialize standard Pinecone client for index creation
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        # Initialize gRPC client for faster uploads
        self.pc_grpc = PineconeGRPC(api_key=config.pinecone_api_key)

    def process_batch(self, batch: List[Document], index, start_idx: int) -> None:
        """Process and upsert a batch of documents."""
        try:
            # Get texts and metadatas
            texts = [doc.content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # Create embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Create vectors
            vectors = [
                (f"doc_{start_idx+j}", embedding, metadata)
                for j, (embedding, metadata) in enumerate(zip(embeddings, metadatas))
            ]
            
            # Upsert to Pinecone with async request
            return index.upsert(vectors=vectors, async_req=True)
        except Exception as e:
            print(f"Error processing batch starting at {start_idx}: {str(e)}")
            return None

    def upload(self, documents: List[Document], index_name: str):
        """Upload documents to Pinecone index using parallel processing."""
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.config.pinecone_region
                )
            )

        # Get gRPC index for faster uploads
        index = self.pc_grpc.Index(index_name)
        
        print(f"Uploading {len(documents)} documents to Pinecone index: {index_name}")
        
        # Calculate optimal batch size based on vector dimension and metadata
        # Using conservative batch size of 200 as per Pinecone docs recommendation
        batch_size = 200
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # Process batches in parallel with progress bar
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, batch in enumerate(batches):
                future = executor.submit(self.process_batch, batch, index, i * batch_size)
                futures.append(future)
            
            # Wait for results with progress bar
            with tqdm(total=len(futures), desc="Uploading batches") as pbar:
                for future in as_completed(futures):
                    try:
                        # Get the result to check for any errors
                        result = future.result()
                        if result:
                            result.result()  # Wait for the async request to complete
                    except Exception as e:
                        print(f"Error in batch upload: {str(e)}")
                    pbar.update(1)
        
        print(f"Success: Uploaded all documents to index {index_name}")