# Python 
venv is virtual environment

## Use venv: activate the virtual environment, you see (venv) haraldhiemstra@Haralds-MacBook-Pro python_ai %

source venv/bin/activate:

## update requirements.txt based on the venv/lib libraries
pip3 freeze > requirements.txt: 

# Ai Flow
1. Original text
2. Text splitter splits the text in several chunks of text
3. Chunks are converted into embeddings (numerical representation) using an embedding model
4. Embeddings are stored into a vector database
5. User inputs a query and is converted into an embedding
6. Perform similarity search between the query embedding and the embeddings in the vector database

# OLLAMA
Make sure ollama is run at http://localhost:11434


# Dunder methods and overloading:
# __init__(): # you can override functions that python calls under the hood, for example  
