import streamlit as st
import os
from indexing import load_documents, split_docs, save_to_db
from query_data import response
UPLOAD_FOLDER = "../Docs"


def upload_file(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    uploaded_file = st.file_uploader("Chọn tài liệu muốn hỏi đáp", type=["txt", "pdf"])
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

def get_question():
    question = st.text_input("Nhập câu hỏi của bạn")
    return question

def save_docs_to_db():
    docs = load_documents(UPLOAD_FOLDER)
    chunks = split_docs(docs)
    save_to_db(chunks)

def main():
    upload_file(UPLOAD_FOLDER)
    question = get_question()
    answer = response(question)
    st.write(answer)


if __name__ == "__main__":
    main()