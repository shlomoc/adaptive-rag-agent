from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    """
    Decide whether to generate an answer or perform a web search based on the current state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        str: Either WEBSEARCH or GENERATE, indicating the next step in the workflow.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    print("---DECISION: GENERATE---")
    return GENERATE


def grade_generation(state: GraphState) -> str:
    """
    Grade the generated answer for hallucinations and relevance to the question.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        str: Either "useful", "not useful", or "not supported", indicating the quality of the generation.
    """
    print("---CHECK HALLUCINATIONS---")
    question, documents, generation = (
        state["question"],
        state["documents"],
        state["generation"],
    )

    if hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    ).binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        if answer_grader.invoke(
            {"question": question, "generation": generation}
        ).binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---\n")
            return "useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---\n")
        return "re-search"
    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---\n")
    return "hallucination"


def route_question(state: GraphState) -> str:
    """
    Determine whether to route the question to web search or RAG.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        str: Either WEBSEARCH or RETRIEVE, indicating the next step in the workflow.
    """
    print("---ROUTE QUESTION---")
    source: RouteQuery = question_router.invoke({"question": state["question"]})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    print("---ROUTE QUESTION TO RAG---")
    return "retrieve"


def create_workflow() -> StateGraph:
    """
    Create and return the workflow graph.

    Returns:
        StateGraph: The compiled workflow graph.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    for node, func in [
        (RETRIEVE, retrieve),
        (GRADE_DOCUMENTS, grade_documents),
        (GENERATE, generate),
        (WEBSEARCH, web_search),
    ]:
        workflow.add_node(node, func)

    # Set entry point and edges
    workflow.set_conditional_entry_point(
        route_question, {"websearch": WEBSEARCH, "retrieve": RETRIEVE}
    )
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
    )
    workflow.add_conditional_edges(
        GENERATE,
        grade_generation,
        {
            "hallucination": GENERATE,
            "useful": END,
            "re-search": WEBSEARCH,
        },
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GENERATE, END)

    return workflow


def create_graph():
    """
    Main function to create the workflow and generate a graph visualization.
    """
    # Create and compile the workflow
    app = create_workflow().compile()

    # Generate graph visualization
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
