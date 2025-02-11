from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from langchain_postgres import PGVector
from sqlalchemy import create_engine


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
PROMPT_TEMPLATE = """
Answer the question based only on the following context. If you don't know the answer, just say that you don't know.:

{context}

---

Answer the question based on the above context: {question}
"""

def connect_to_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    connection = "postgresql+psycopg://postgres:020104@localhost:5432/Vector"
    engine = create_engine(connection, future=True)
    db = PGVector(
        embeddings=embeddings,
        collection_name="Vector",
        connection=engine
    )
    return db
def response(question):
    db = connect_to_db()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    results = db.similarity_search_with_score(question, k=5)
    context_text = "\n\n---\n\n".join([result.page_content for result, _score in results])
    prompt = prompt_template.format(context=context_text, question=question)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    response = llm.invoke(prompt)
    return response.content
def main():
    print(response("Hai khả năng cốt lõi giúp phát triển trong nền kinh tế mới là gì?"))

if __name__ == "__main__":
    main()