from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone
import os

load_dotenv()

index_name = "advanced-rag"


def ingest_documents():
    """
    Ingest documents from specified URLs into a Pinecone index.
    """
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Create or connect to an existing index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings are 1536 dimensions
            metric="cosine",
        )

    # Create the vector store using Pinecone
    vectorstore = LangchainPinecone.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings(), index_name=index_name
    )

    print(f"Ingested {len(urls)} URLs into Pinecone index '{index_name}'")


def get_retriever():
    """
    Get a retriever for the Pinecone index.

    Returns:
        A retriever object that can be used to query the Pinecone index.
    """
    # Initialize Pinecone
    Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Connect to the existing index
    vectorstore = LangchainPinecone.from_existing_index(
        index_name, embedding=OpenAIEmbeddings()
    )

    # Create a retriever from the Pinecone vector store
    return vectorstore.as_retriever()


if __name__ == "__main__":
    ingest_documents()
