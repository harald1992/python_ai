from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


_rag_prompt_template = """Answer the question below only using the following context:
            {context}
            
            Question: {question}
            """


# A chain refers to a sequence of steps (using models) that are executed in order.
# For example: Tokenizing, embedding, querying the vectordb and generating the response.
def generate_ollama_response(retriever, llm, question):
    prompt = ChatPromptTemplate.from_template(_rag_prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response

