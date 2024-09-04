from typing import Any, Dict

from graph.state import GraphState
from ingestion import get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents based on the given question.

    This function performs the following steps:
    1. Extracts the question from the current state.
    2. Obtains a retriever object from the Pinecone index.
    3. Uses the retriever to find relevant documents for the question.

    Args:
        state (GraphState): The current state of the graph, containing the question.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents and the original question.
            The dictionary has the following structure:
            {
                "documents": List of retrieved documents,
                "question": The original question string
            }
    """
    print("---RETRIEVE---")
    question = state["question"]

    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
