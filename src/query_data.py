
from dotenv import load_dotenv
import os
from openai import OpenAI

import psycopg2


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def connect_to_db():
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    # connection = "postgresql+psycopg://postgres:020104@localhost:5432/Vector"
    # engine = create_engine(connection, future=True)
    # db = PGVector(
    #     embeddings=embeddings,
    #     collection_name="Vector",
    #     connection=engine
    # )
    # return db
    connection =  psycopg2.connect(dbname="Vector", user="postgres", password="020104", host="localhost", port="5432") 
    cursor = connection.cursor() 
    return cursor  
def response(question):
    db = connect_to_db()
    query_embedding = client.embeddings.create(input=question, model="text-embedding-3-small")
    db.execute(f"SELECT * FROM embeddings ORDER BY vector <=> '{query_embedding.data[0].embedding}' LIMIT 5")
    results = db.fetchall()
    context = []
    for result in results:
        context.append(result[2])
    PROMPT_TEMPLATE = f"""
    Answer the question based only on the following context. If you don't know the answer, just say that you don't know.:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE
            }
        ]
    )
    return completion.choices[0].message.content
    
def main():
    print(response("Hai khả năng cốt lõi giúp phát triển trong nền kinh tế mới là gì?"))

if __name__ == "__main__":
    main()