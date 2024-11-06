# Git to RAG

A tool for creating and managing embeddings of code repositories in Pinecone vector database. This tool helps in processing, chunking, and embedding code repositories for efficient semantic search and analysis.

## Features

- 🔄 Efficient repository cloning with caching
- 📝 Smart code chunking with language-specific handling
- 🔍 Intelligent text encoding detection
- 📊 Parallel processing of embeddings
- 🚀 Batch uploading to Pinecone
- 🎨 Colored logging for better visibility
- ⚡ Progress tracking with progress bars

## Prerequisites

- Python 3.8+
- Git
- OpenAI API key
- Pinecone API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/elanrad1/.git
cd Git-to-RAG
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=your_pinecone_region
```

## Usage

1. Configure your repository settings in `main.py`:
```python
repo_url = "https://github.com/username/repository.git"
target_folder = "path/to/folder"  # Optional
model = "gpt-3.5-turbo"  # Model for tokenization
index_name = "your-index-name"
```

2. Run the embedder:
```bash
python main.py
```

The tool will:
- Clone the repository
- Process and chunk the code files
- Create embeddings using OpenAI
- Store vectors in Pinecone
- Cache results for future runs

## Architecture

- `src/repo_cloner.py`: Handles repository cloning and caching
- `src/chunker.py`: Processes files into optimized chunks
- `src/pinecone_uploader.py`: Manages vector embeddings and uploads
- `src/config.py`: Centralizes configuration management
- `src/utils.py`: Provides utility functions and helpers

## Performance Optimizations

- Parallel processing for batch uploads using ThreadPoolExecutor
- gRPC implementation for faster Pinecone operations
- Efficient caching system for incremental updates
- Smart chunking with language-specific separators
- Optimized batch sizes for vector uploads (200 vectors per batch)
- Async requests for better throughput
- Progress bars for monitoring long operations

## Caching System

The tool implements a sophisticated caching system that:
- Stores processed chunks in JSON format
- Uses MD5 hashing for file change detection
- Maintains separate metadata for repos and chunks
- Implements safe directory operations with retries
- Only reprocesses modified files
- Preserves file metadata across runs

## File Processing

Supports multiple file types with specialized handling:
```python
default_extensions = {
    '.py', '.js', '.java', '.cpp', '.c', '.h', 
    '.cs', '.php', '.rb', '.go', '.txt', '.md',
    '.rst', '.json', '.yml', '.yaml', '.toml',
    '.ini', '.cfg', '.conf'
}
```

Features:
- Automatic encoding detection
- Language-specific chunking strategies
- Code-aware text splitting
- Token counting using model tokenizer
- Metadata preservation

## Vector Upload System

Implements high-performance vector uploads:
- Uses Pinecone gRPC client for speed
- Parallel batch processing
- Async requests with error handling
- Progress tracking with tqdm
- Automatic index creation if needed
- Configurable batch sizes and workers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for embeddings
- Pinecone for vector storage
- LangChain for utilities
