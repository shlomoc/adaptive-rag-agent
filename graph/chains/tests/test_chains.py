from dotenv import load_dotenv
from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.router import question_router, RouteQuery
from ingestion import get_retriever  # Changed this line

load_dotenv()


def get_test_documents(question: str):
    """
    Retrieve test documents for a given question.

    Args:
        question (str): The question to retrieve documents for.

    Returns:
        tuple: A tuple containing the full document list and the content of the second document.
    """
    retriever = get_retriever()  # Get the retriever
    docs = retriever.invoke(question)
    return docs, docs[1].page_content


def test_retrieval_grader_answer_yes():
    """Test the retrieval grader with a relevant document."""
    question = "agent memory"
    _, doc_txt = get_test_documents(question)

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no():
    """Test the retrieval grader with an irrelevant document."""
    question = "agent memory"
    _, doc_txt = get_test_documents(question)

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain():
    """Test the generation chain."""
    question = "agent memory"
    retriever = get_retriever()  # Get the retriever
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
    assert generation, "Generation should not be empty"


def test_hallucination_grader_answer_yes():
    """Test the hallucination grader with a non-hallucinated answer."""
    question = "agent memory"
    retriever = get_retriever()  # Get the retriever
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no():
    """Test the hallucination grader with a hallucinated answer."""
    question = "agent memory"
    retriever = get_retriever()  # Get the retriever
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore():
    """Test the question router for a question that should be routed to the vectorstore."""
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch():
    """Test the question router for a question that should be routed to web search."""
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
