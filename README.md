# NextJs Documentation Assistant (RAG Chatbot)

## Overview

In this project, we will implement a chatbot ([RAG](https://www.promptingguide.ai/techniques/rag) model) that can answer
questions about Next.js documentation.

1. **Documentation Acquisition:** Downloading HTML content from
   the [Next.js Official Documentation](https://nextjs.org/docs).
2. **HTML Scrapping:** Extracting critical data, focusing on the `article` tag from each page.
3. **Data Processing:** Tokenizing and vectorizing the collected information.
4. **Data Indexing:** Storing the processed data in a [Pinecone](https://www.promptingguide.ai/techniques/pinecone)
   index for efficient retrieval.
5. **Chatbot Creation:** Using [LangChain](https://www.langchain.com/) in conjunction
   with [OpenAI models](https://platform.openai.com/docs/models) and the Pinecone index to develop a responsive chatbot.

## Getting Started

## Get started

### Prerequisites

- [Python 3.10.13](https://www.python.org/downloads/)
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [Pinecone API key](https://app.pinecone.io/organizations/-Nl-1ULxOo96JUER84X4/projects/gcp-starter:96uoss8/keys)

### Setup the project

1. Create a python virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Duplicate .env.template to create .env..
   > Remember to populate it with your **OpenAI API key**, **Pinecone API key**, and **Pinecone environment name**.

5. To download the sources, run the following command in your terminal:

```bash
python download_sources.py
```

6. To vectorize the sources, run the following command in your terminal:

```bash
python vectorize_sources.py
```

7. To start the assistant, run the following command in your terminal:

```bash
streamlit run main.py
```