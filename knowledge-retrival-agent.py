import os
import json

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.llms.minimax import Minimax
from langchain_community.document_loaders import (
    DirectoryLoader
)
from langchain_ollama import (OllamaEmbeddings, OllamaLLM)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.6:35b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")

def main():

    loader = DirectoryLoader(
        path="./docs",
        glob="*.txt"
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

    llm = OllamaLLM(model=LLM_MODEL)
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the following context to answer the question. If you don't know the answer, say you don't know.
         Context: {context}"""),
        ("user", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_separator="\n\n"
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    query = "When Eiffel Tower was built?"

    result = retrieval_chain.invoke({
        "input": query
    })

    print(f"Answer: {result['answer']}")

    print(f"Sources Len: {len(result['context'])}")

if __name__ == "__main__":
    main()

