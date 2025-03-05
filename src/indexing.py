
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import os
from pypdf import PdfReader
    

pdf_folder = "../Docs"
CHROMA_PATH = "../chroma"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
def load_documents(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            reader = PdfReader(pdf_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            documents.append({"text": text})
    return documents

def split_docs(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# def save_to_db(chunks):
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
#     db = Chroma.from_documents(
#         chunks, embeddings, persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db

def save_to_db(docs):
    conn = psycopg2.connect(
        dbname="Vector",
        user="postgres",
        password="020104",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    register_vector(conn)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            vector vector(1536),  
            content TEXT NOT NULL,
            dimension INT NOT NULL 
        )
    """)
    conn.commit()
    texts = [doc["text"] for doc in docs]
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [item.embedding for item in response.data]
    for doc, embedding in zip(docs, embeddings):
        doc["embedding"] = embedding
    
    insert_query = """
    INSERT INTO embeddings (vector, content, dimension)
    VALUES %s
    """
    data_to_insert = [
    (embedding, doc["text"], len(embedding)) for doc, embedding in zip(docs, embeddings)
    ]
    execute_values(cur, insert_query, data_to_insert)
    conn.commit()
    cur.execute("CREATE INDEX ON embeddings USING hnsw (vector vector_cosine_ops)")
    conn.commit()
def main():
    docs = load_documents(pdf_folder)
    chunked_documents = []
    for doc in docs:
        chunks = split_docs(doc['text'])
        print("==== Splitting docs into chunks ====")
        for i, chunk in enumerate(chunks):
            chunked_documents.append({id: i, "text": chunk})
    save_to_db(chunked_documents)
    


if __name__=="__main__":
    main()