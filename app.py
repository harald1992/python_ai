# Imports

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama


# from typing import List, Tuple, Dict, Any, Optional
# from langchain_ollama.llms import OllamaLLM

# Local Imports
from initialize_vector_db_module import initialize_vector_db
from ollama_retriever_module import create_retriever
from generate_ollama_response import generate_ollama_response


def ask_user_input():
    while True:
        user_input = input("Please ask your question for the AI:\n")
        response = generate_ollama_response(retriever, llm, user_input)
        print(response)

# Starting the scripts
vector_db = initialize_vector_db()
retriever, llm = create_retriever(vector_db)
ask_user_input()


# TODO: use uv for packages and local usage of lib versions/package lock: https://astral.sh/blog/uv
