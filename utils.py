import hashlib
import json
from typing import Any
from bs4 import BeautifulSoup
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from tqdm import tqdm
from tiktoken import Encoding


def generate_uid_from_str(text: str, length: int = 12) -> str:
    m = hashlib.md5()
    m.update(text.encode("utf-8"))
    uid = m.hexdigest()[:length]
    return uid


def save_as_jsonl(docs: list[Any], filename: str):
    with open(filename, "w") as file:
        for doc in tqdm(docs):
            file.write(json.dumps(doc) + "\n")


def load_from_jsonl(filename: str) -> list[Any]:
    with open(filename, "r") as file:
        return [json.loads(line) for line in tqdm(file)]


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

    for document in documents:
        soup = BeautifulSoup(document.page_content, "html.parser")
        article_soup = soup.find("article")

        if article_soup is not None:
            document.page_content = article_soup.get_text(strip=True, separator="\n\n")
            document.metadata["source"] = document.metadata["source"].replace(
                "rtdocs/", "https://"
            )
            document.metadata["source"] = document.metadata["source"].replace(
                ".html", ""
            )
            result.append(document)

    return result


def tiktoken_len(text, tokenizer: Encoding):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)
