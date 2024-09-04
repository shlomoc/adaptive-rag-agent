from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from graph.consts import MODEL_NAME

# Constants
SYSTEM_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""


class GradeAnswer(BaseModel):
    """Binary score for whether an answer addresses the question."""

    binary_score: bool = Field(
        description="Answer addresses the question: True for 'yes', False for 'no'"
    )


def create_answer_grader() -> RunnableSequence:
    """Create and return an answer grader sequence."""
    llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "User question:\n\n{question}\n\nLLM generation: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader


# Create the answer grader
answer_grader = create_answer_grader()
