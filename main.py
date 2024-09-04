import streamlit as st
from dotenv import load_dotenv
from graph.graph import create_workflow, create_graph
from pprint import pprint
import langchain
import sys

# Load environment variables
load_dotenv()

# Enable langchain debugging
# langchain.debug = True

# Generate the graph visualization once at the top level
graph = create_graph()


def chat():
    """Main chat function that handles the Streamlit UI and interaction with the RAG system."""
    st.markdown("# Advanced RAG Chat")
    st.markdown(
        "<h4 style='font-weight: normal; margin-top: -15px;'>about</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "[https://lilianweng.github.io/posts/](https://lilianweng.github.io/posts/)"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your question here ..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            process_user_input(prompt)


def process_user_input(prompt):
    """Process user input and generate response."""
    with st.spinner("Generating response.."):
        app = create_workflow().compile()

        # Stream the execution and print "Finished running" for each node
        for output in app.stream({"question": prompt}):
            for key in output:
                pprint(f"Finished running: {key}")

        # Get the final result
        result = app.invoke({"question": prompt})

        st.markdown(result["generation"])
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["generation"]}
        )


def main():
    """Run the workflow without UI for debugging purposes."""
    app = create_workflow().compile()

    # Example question for debugging
    debug_question = "What is RAG?"

    print(f"Debug question: {debug_question}")
    print("Running workflow...")

    # Stream the execution and print "Finished running" for each node
    for output in app.stream({"question": debug_question}):
        for key in output:
            print(f"Finished running: {key}")

    # Get the final result
    result = app.invoke({"question": debug_question})

    print("\nFinal result:")
    pprint(result)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        main()
    else:
        chat()
