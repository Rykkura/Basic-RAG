from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

CHROMA_PATH = "chroma"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
PROMPT_TEMPLATE = """
Answer the question based only on the following context. If you don't know the answer, just say that you don't know.:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    query = "Hai khả năng cốt lõi giúp phát triển trong nền kinh tế mới là gì?"

    results = db.similarity_search_with_score(query, k=5)
    context_text = "\n\n---\n\n".join([result.page_content for result, _score in results])
    prompt = prompt_template.format(context=context_text, question=query)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    response = llm.invoke(prompt)
    print(response.content)

if __name__ == "__main__":
    main()