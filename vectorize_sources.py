import pinecone
import tiktoken
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

from environements import (
    llm_model_name,
    sources_folder_path,
    openai_api_key,
    pinecone_api_key,
)
from utils import (
    load_docs,
    tiktoken_len,
    context_window_size_by_model,
    dimension_by_embedding_model,
    pinecone_index_name,
    embedding_model_name,
    pinecone_environment,
)

load_dotenv()


def main():
    print("📢 script started !")

    print("🛠️ Loading sources...")

    documents = load_docs(sources_folder_path, glob="nextjs.org/docs/app/**/*.html")

    print(f"✅ Loaded {len(documents)} sources.")

    tokenizer = tiktoken.encoding_for_model(llm_model_name)
    context_window_size = context_window_size_by_model[llm_model_name]

    print(f"ℹ️ using token encoding : {tokenizer.name} for {llm_model_name} llm model")
    print(f"ℹ️ Llm model context window size: {context_window_size}")

    token_counts = [
        tiktoken_len(document.page_content, tokenizer) for document in documents
    ]

    print("ℹ️ Token counts for sources:")
    print(f"  - Min: {min(token_counts)}")
    print(f"  - Avg: {int(sum(token_counts) / len(token_counts))}")
    print(f"  - Max: {max(token_counts)}")

    print("🛠️ Creating chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,  # number of tokens overlap between chunks
        length_function=lambda text: tiktoken_len(text, tokenizer),
        separators=["\n\n", "\n", " ", ""],
    )

    tokenized_documents = text_splitter.split_documents(documents)

    print(f"✅ Created {len(tokenized_documents)} chunks.")

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name, openai_api_key=openai_api_key
    )

    print("🛠️ Getting pinecone index...")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    print(f"🛠️ Checking if index {pinecone_index_name} exists...")
    if pinecone_index_name in pinecone.list_indexes():
        print(f"✅ Index {pinecone_index_name} exists.")

        print(f"🛠️ Deleting index {pinecone_index_name}...")

        pinecone.delete_index(pinecone_index_name)

        print(f"✅ Index {pinecone_index_name} deleted.")

    print("🛠️ Creating pinecone index...")

    pinecone.create_index(
        name=pinecone_index_name,
        metric="cosine",
        dimension=dimension_by_embedding_model[embedding_model_name],
    )

    print(f"✅ Index {pinecone_index_name} created.")

    print("🛠️ Uploading documents to pinecone...")

    Pinecone.from_documents(documents, embeddings, index_name=pinecone_index_name)

    print(f"✅ {len(documents)} documents uploaded to pinecone.")


if __name__ == "__main__":
    main()
