from typing import Any

import streamlit as st

from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
from chatbot import init_conversational_retrieval_streaming_chain

load_dotenv()

USER_AVATAR = "👱‍♂️"
ASSISTANT_AVATAR = "🤖"


def main():
    st.set_page_config(page_title="NextJs Assistant", page_icon="🤖")

    st.title("NextJs Documentation Assistant")

    display_sidebar()

    display_chat_messages_history()

    if "agent" not in st.session_state:
        initialise_agent()

    if user_input := st.chat_input("Your Message ?"):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            assistant_message_placeholder = st.empty()

            response = get_bot_response(
                user_input,
                [StreamlitCallbackHandler(assistant_message_placeholder)],
            )

            assistant_message_placeholder.markdown(response)


def initialise_agent(callbacks: list[Any] = None) -> None:
    st.session_state.agent = init_conversational_retrieval_streaming_chain(
        callbacks=callbacks
    )


def get_bot_response(user_input: str, callbacks: list[Any] = None) -> Any:
    response = st.session_state.agent(
        {"question": user_input},
        callbacks=callbacks,
    )

    sources_md = "\n".join(
        [
            f"- *[{source_document.metadata['title']}]({source_document.metadata['source']})*"
            for source_document in response["source_documents"]
        ]
    )

    answer = f"{response['answer']}\n\n---\n### *Sources:*\n{sources_md}"

    st.session_state.agent.memory.chat_memory.messages[-1].content = answer
    return answer


def display_chat_messages_history() -> None:
    if "agent" not in st.session_state:
        return

    for message in st.session_state.agent.memory.chat_memory.messages:
        with st.chat_message(
            message.type,
            avatar=ASSISTANT_AVATAR if message.type == "ai" else USER_AVATAR,
        ):
            st.markdown(message.content)


def display_sidebar() -> None:
    with st.sidebar:
        with st.form(key="my_form"):
            if st.form_submit_button(label="Reset"):
                st.session_state.agent.memory.chat_memory.clear()


if __name__ == "__main__":
    main()
