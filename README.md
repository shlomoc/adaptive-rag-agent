
# Advanced RAG control flow with LangGraph🦜🕸:

Implementation of Corrective RAG, Self-RAG & Adaptive RAG tailored towards developers and production-oriented applications for learning LangGraph🦜🕸️.

This repository contains a refactored version of the original [LangChain's Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain),

See Original YouTube video:[Advance RAG control flow with Mistral and LangChain](https://www.youtube.com/watch?v=sgnrL7yo1TE)
by Sophia Young from Mistral & Lance Martin from LangChain from LangChain


## Features

- **Replaced Chroma db vectorstore with Pinecone**
- **Added Streamlit UI chatbot**
- **Refactored Notebooks**: The original LangChain notebooks have been refactored to enhance readability, maintainability, and usability for developers.
- **Production-Oriented**: The codebase is designed with a focus on production readiness, allowing developers to seamlessly transition from experimentation to deployment.
- **Test Coverage**: Test coverage ensures the reliability and stability of the application, enabling developers to validate their implementations effectively.
- **Documentation**: Detailed documentation and branches guides developers through setting up the environment, understanding the codebase, and utilizing LangGraph effectively.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`PYTHONPATH=/{YOUR_PATH_TO_PROJECT}/adaptive-rag-chatbot`

`OPENAI_API_KEY`
`TAVILY_API_KEY`
`PINECONE_API_KEY`


## Run Locally

Clone the project

```bash
  git clone https://github.com/shlomoc/adaptive-rag-chatbot.git
```

Go to the project directory

```bash
  cd adaptive-rag-chatbot
```

Install dependencies

```bash
  poetry install
```

Run the Streamlit app

```bash
  streamlit run main.py
```


## Running Tests

To run tests, run the following command

```bash
  poetry run pytest . -s -v
```

## Testing from command line (no UI)

```bash
run main.py --debug
```

  
## Acknowledgements
* [Langgraph course](https://www.udemy.com/course/langgraph/?referralCode=FEA50E8CBA24ECD48212) by Eden Marco
* Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain) By Sophia Young from Mistral & Lance Martin from LangChain



