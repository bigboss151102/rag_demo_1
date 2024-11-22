from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import psycopg2
from psycopg2 import OperationalError
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

from dotenv import load_dotenv
import os
import json

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
postgres_user = os.getenv('POSTGRES_USER')
postgres_password = os.getenv('POSTGRES_PASSWORD')
postgres_db = os.getenv('POSTGRES_DB')

    
loader = JSONLoader(
    file_path="/Users/ttcenter/AvocaEdu/demo_vector_store/data.json",
    jq_schema="""
            .data[] | 
            { 
                name,
                price,
                html_content,
                total_sold,
                promotion_price,
                available_quantity: .goods.available_quantity,
                brand_name: .brand.name,
                branch_description: .branch.description,
                category_name: .category.name,
                quantity_warehouse: (.warehouse_good_items[0].quantity // 0)
            }
        """,
        text_content=False)

documents = loader.load()

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=openai_api_key)


text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# flattened_docs = [Document(page_content=doc.page_content) for doc in documents if doc.page_content]
flattened_docs = [doc for doc in documents if doc.page_content]
initial_chunks = text_splitter.split_documents(flattened_docs)

semantic_chunker = SemanticChunker(embeddings=embeddings)
chunks = semantic_chunker.split_documents(initial_chunks)

print("Chunksssss: ", chunks)

connection_string = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@localhost:5435/{postgres_db}"

def create_connection():
    try:
        connection = psycopg2.connect(
            user=postgres_user,
            password=postgres_password,
            host="localhost", 
            port="5435",
            database=postgres_db
        )
        return connection
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None

connection = create_connection()

if connection is not None:
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="products",
        connection_string=connection_string,
        pre_delete_collection=True,
    )
else:
    print("Failed to connect to the database.")

