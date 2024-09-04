from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from graph.consts import MODEL_NAME

# Constants
SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts: True for 'yes', False for 'no'"
    )


def create_hallucination_grader() -> RunnableSequence:
    """Create and return a hallucination grader sequence."""
    llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
        ]
    )

    return hallucination_prompt | structured_llm_grader


# Create the hallucination grader
hallucination_grader = create_hallucination_grader()
