import hashlib

import pinecone
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain_core.documents import Document
from tiktoken import Encoding
from tqdm import tqdm

from environements import (
    embedding_model_name,
    openai_api_key,
    pinecone_environment,
    pinecone_api_key,
    pinecone_index_name,
    vector_text_field,
    sources_folder_path,
)

load_dotenv()


context_window_size_by_model = {
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo": 4096,
    "gpt-4-1106-preview": 128000,
}

dimension_by_embedding_model = {"text-embedding-ada-002": 1536}


def generate_uid_from_str(text: str, length: int = 12) -> str:
    m = hashlib.md5()
    m.update(text.encode("utf-8"))
    uid = m.hexdigest()[:length]
    return uid


def load_docs(
    directory: str,
    glob: str = "**/*.html",
    show_progress: bool = True,
    sample_size: int = 0,
) -> list[Document]:
    result = []

    loader = DirectoryLoader(
        directory,
        glob=glob,
        show_progress=show_progress,
        sample_size=sample_size,
        loader_cls=TextLoader,
    )

    documents = loader.load()

    for document in tqdm(documents):
        soup = BeautifulSoup(document.page_content, "html.parser")
        article_soup = soup.find("article")
        title_soup = soup.find("title")

        if article_soup is not None:
            document.page_content = article_soup.get_text(strip=True, separator="\n\n")
            document.metadata["source"] = document.metadata["source"].replace(
                f"{sources_folder_path}/", "https://"
            )
            document.metadata["source"] = document.metadata["source"].replace(
                ".html", ""
            )

            document.metadata["uid"] = generate_uid_from_str(
                document.metadata["source"]
            )

            if title_soup is not None:
                document.metadata["title"] = title_soup.get_text(
                    strip=True, separator="\n\n"
                )

            result.append(document)

    return result


def tiktoken_len(text, tokenizer: Encoding) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_pinecone_index() -> Pinecone:
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name, openai_api_key=openai_api_key
    )

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    index = pinecone.Index(pinecone_index_name)

    return Pinecone(index, embeddings, vector_text_field)
