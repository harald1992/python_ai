import os
import ssl

import chromadb
import nltk
import uuid
from chromadb import EmbeddingFunction
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def initialize_vector_db():
    _initialize_punkt()
    chunks = _create_chunks_from_pdf()
    return _create_vector_db(chunks)


# Punkt is data package used by Natural Language Toolkit (NLTK) to tokenize text (splitting text)
def _initialize_punkt():
    print("Initializing Punkt...")
    # Set the NLTK_DATA environment variable
    os.environ['NLTK_DATA'] = 'venv/nltk_data'

    # Bypass SSL verification and download 'punkt' data
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Check if 'punkt' is already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt is already downloaded.")
    except LookupError:
        nltk.download('punkt', download_dir='venv/nltk_data')

    # Check if 'punkt_tab' is already downloaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("Punkt_tab is already downloaded.")
    except LookupError:
        nltk.download('punkt_tab', download_dir='venv/nltk_data')

    # Check if 'averaged_perceptron_tagger_eng' is already downloaded
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        print("Averaged_perceptron_tagger_eng is already downloaded.")
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', download_dir='venv/nltk_data')


def _create_chunks_from_pdf():
    local_path = "pdf/Hypotheek-verhogen-Ontdek-de-mogelijkheden-Rabobank.pdf"
    if local_path:
        loader = UnstructuredPDFLoader(file_path=local_path)
        data = loader.load()
        print(f"PDF loaded successfully in memory: {local_path}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)  # can set smaller chunk size probably like 500 for smaller data sets.
        chunks = text_splitter.split_documents(data)
        print(f"PDF split into {len(chunks)} chunks")

        return chunks
    else:
        print(f"Pdf upload failed for {local_path}")


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.model = OllamaEmbeddings(model=model_name)

    def __call__(self, texts):
        print(f"Call: {texts}")
        return self.model.embed_documents(texts)

    def embed(self, texts):
        print(f"Embed: {texts}")
        return self.model.embed_documents(texts)


def _create_vector_db(chunks):
    print("Creating Vector DB...")

    httpClient = chromadb.HttpClient(host="localhost", port=8001, settings=Settings(allow_reset=True))

    if httpClient.get_collection("my-rag-collection").count()> 0:
        httpClient.delete_collection("my-rag-collection")

    embedding_function = OllamaEmbeddingFunction(model_name="llama3.1")

    collection = httpClient.get_or_create_collection(
        name="my-rag-collection",
        embedding_function=embedding_function,

        # embedding_function = embedding_function
        # all-MiniLM-L6-v2 is the default embedding mode.
    )
    print(f"Collection: {collection}")
    print(f"Collection Name: {collection.name}")
    print(f"Collection Metadata: {collection.metadata}")

    if collection.count() > 0:
        print("Document already exists. Skipping...")
    else:
        print("Loading document...")
        for doc in chunks:
            collection.add(
                ids=[str(uuid.uuid1())],
                metadatas=doc.metadata,
                documents=doc.page_content,
            )

    # # collection.add_document("my-rag-collection", "my-rag-document", chunks[0])
    #     collection.add(
    #         ids=[str(uuid.uuid1())],
    #         documents=chunks,
    #         # embeddings=embeddings
    #     )

    vector_db = Chroma(
        client=httpClient,
        collection_name="my-rag-collection",
        embedding_function=OllamaEmbeddings(model="llama3.1"),
    )

    print(f"Vector DB created successfully: {vector_db}")
    return vector_db

