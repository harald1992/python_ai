# Python 
venv is virtual environment

# Ai Flow
1. Original text
2. Text splitter splits the text in several chunks of text
3. Chunks are converted into embeddings (numerical representation) using an embedding model
4. Embeddings are stored into a vector database
5. User inputs a query and is converted into an embedding
6. Perform similarity search between the query embedding and the embeddings in the vector database
