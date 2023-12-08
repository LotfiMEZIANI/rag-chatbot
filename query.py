from langchain_core.callbacks import StreamingStdOutCallbackHandler

from chatbot import init_conversational_retrieval_streaming_chain


def main():
    chain = init_conversational_retrieval_streaming_chain()
    query = "Explain me middle ware in short sentence ?"

    result = chain({"question": query}, callbacks=[StreamingStdOutCallbackHandler()])

    sources = (
        [
            source_document.metadata["source"]
            for source_document in result["source_documents"]
        ],
    )

    sources_md = "\n".join([f"- {source}" for source in sources])

    print(result)


if __name__ == "__main__":
    main()
