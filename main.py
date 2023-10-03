import streamlit as st
from langchain_functions import (
    get_pdf_text,
    get_pdf_chunks,
    get_vector_store,
    get_conversation_chain,
)


def disable():
    st.session_state["disabled"] = True


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("bot"):
                st.write(message.content)


def main():
    # check for conversation state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "disabled" not in st.session_state:
        st.session_state["disabled"] = False

    st.set_page_config(page_title="Chat with Documents", page_icon=":books:")
    st.header("Chat with your Documents")

    user_question = st.chat_input(
        "Ask a question about your documents", disabled=st.session_state.disabled
    )
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        files = st.file_uploader(
            "upload files here", accept_multiple_files=True, type=["pdf"]
        )
        if st.button("Upload", on_click=disable):
            with st.spinner("uploading"):
                # get pdf text
                raw_text = get_pdf_text(files)
                # get text chuncks
                text_chunks = get_pdf_chunks(raw_text)
                # create vector store
                vector_store = get_vector_store(text_chunks)
                # create chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
