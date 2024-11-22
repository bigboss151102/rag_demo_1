import os
from operator import itemgetter
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel

import json

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
postgres_user = os.getenv('POSTGRES_USER')
postgres_password = os.getenv('POSTGRES_PASSWORD')
postgres_db = os.getenv('POSTGRES_DB')


conection_string = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@localhost:5435/{postgres_db}"

vector_store = PGVector(
    collection_name="products",
    connection_string=conection_string,
    embedding_function=OpenAIEmbeddings()
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini', streaming=True)


class RagInput(TypedDict):
    question: str

retriever = vector_store.as_retriever()

final_chain = (
    RunnableParallel(
        context=(itemgetter("question") | vector_store.as_retriever()),
        question=itemgetter("question")
    ) |
    RunnableParallel(
        answer=(ANSWER_PROMPT | llm),
        docs=itemgetter("context")
    )
)


query = {
    "question": "Giá của Khăn Ướt SmartAngel Không Mùi là bao nhiêu?"
}
result = final_chain.invoke(query)

for doc in result['docs']:
    raw_content = doc.page_content

    try:
        decoded_content = bytes(raw_content, 'utf-8').decode('unicode_escape')
    except json.JSONDecodeError:
        print("Error:", raw_content)