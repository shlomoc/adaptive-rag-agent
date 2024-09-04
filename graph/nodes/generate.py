from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate an answer based on the given question and retrieved documents.

    Args:
        state (GraphState): The current state of the graph, containing the question and documents.

    Returns:
        Dict[str, Any]: A dictionary containing the documents, question, and generated answer.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
