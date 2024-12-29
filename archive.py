

# def query_ollama(question):x§
#     print("Querying Ollama with question: ", question)
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "llama3.1",
#         "prompt": question,
#         "stream": True}
#     headers = {"Content-Type": "application/json"}
#
#     response = requests.post(url, json=payload, headers=headers)
#
#     print(response)
#     if response.status_code == 200:
#         for line in response.iter_lines(delimiter=b"\n", chunk_size=1):
#             if line:
#                 print(json.loads(line.decode('utf-8'))["response"], end="")
#
#         return response
#     else:
#         raise Exception("Failed to query Ollama")





# def prompt_ollama(user_question):
#     prompt_template = """Question: {question}
#     Answer: If you don't know the answer for sure, just say I don't know"""
#
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#
#     model = OllamaLLM(model="llama3.1")
#
#     chain = prompt | model
#
#     for chunk in chain.stream({"question": user_question}):
#         yield chunk
#     yield "\n\n"
#
#



#
# vector_db = Chroma.from_documents(
#     documents=chunks,
#     embedding=OllamaEmbeddings(model="llama3.1"),   # or use nomic-embed-text
#     collection_name="my-local-rag",
#     host="chroma-db",
#     port=8000)
