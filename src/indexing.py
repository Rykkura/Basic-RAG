from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_postgres import PGVector
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
    

pdf_folder = "../Docs"
CHROMA_PATH = "../chroma"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_pdf(pdf_folder):
    loader = DirectoryLoader(pdf_folder, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs

def split_docs(docs):
    text__splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n", ". "])
    chunks = text__splitter.split_documents(docs)
    return chunks

# def save_to_db(chunks):
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
#     db = Chroma.from_documents(
#         chunks, embeddings, persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db

def save_to_db(chunks):
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    connection = "postgresql+psycopg://postgres:020104@localhost:5432/Vector"
    engine = create_engine(connection, future=True)
    collection_name = "Vector"
    db = PGVector.from_documents(documents=chunks, embedding=embeddings, collection_name=collection_name, connection=engine)
    return db


def main():
    docs = load_pdf(pdf_folder)
    chunks = split_docs(docs)
    save_to_db(chunks)
    


if __name__=="__main__":
    main()