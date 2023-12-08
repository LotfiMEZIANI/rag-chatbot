from typing import Any

from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from environements import llm_model_name
from utils import get_pinecone_index


def init_retrieval_qa_with_sources_chain() -> RetrievalQAWithSourcesChain:
    vectorstore = get_pinecone_index()

    llm = ChatOpenAI(temperature=0.5, model=llm_model_name)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        reduce_k_below_max_tokens=True,
        memory=memory,
    )


def init_conversational_retrieval_chain() -> BaseConversationalRetrievalChain:
    vectorstore = get_pinecone_index()

    llm = ChatOpenAI(temperature=0.5, model=llm_model_name)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), verbose=True, memory=memory
    )


def init_conversational_retrieval_streaming_chain(
    callbacks: list[Any] = None,
) -> BaseConversationalRetrievalChain:
    vectorstore = get_pinecone_index()

    llm = ChatOpenAI(
        temperature=0.5,
        model=llm_model_name,
        streaming=True,
        callbacks=callbacks,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        verbose=True,
        memory=memory,
        return_source_documents=True,
    )
