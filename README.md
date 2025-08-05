# LCBCHAT
A very basic Go-based all-in-one RAG LLM, based off data scraped from a root URL. Current capabilities:
- Web scraping of all html and pdfs found based on recursive link checks from the root URL
- Processing and chuinking of scraped content
- Loading of chunks into a vector database

Basic prerequisites:
- Ollama (nomic-embed-text model for embedding)
- VectorDB - Currently using qdrant: `<podman|docker> run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`
