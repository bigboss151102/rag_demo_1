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

from openai import OpenAI

from bs4 import BeautifulSoup

from dotenv import load_dotenv
import os
import json

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
postgres_user = os.getenv('POSTGRES_USER')
postgres_password = os.getenv('POSTGRES_PASSWORD')
postgres_db = os.getenv('POSTGRES_DB')


client = OpenAI(api_key=openai_api_key)


def get_document(json_path):
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
    return documents


def get_content(documents):
    all_content = []
    for document in documents:
        content_dict = json.loads(document.page_content)
        if "html_content" in content_dict:
            soup = BeautifulSoup(content_dict["html_content"], 'html.parser')
            cleaned_text = soup.get_text()
            content_dict["html_content"] = cleaned_text
        all_content.append(content_dict)
    return all_content


def complete_sentense(content_dict):
    system_prompt = "Bạn là một trợ lý ảo viết lại nội dung thông minh.bạn sẽ được cung cấp data kiểu dictionary về một sản phẩm."
    user_prompt = f"""
        Hãy giúp tôi viết thành một câu hoàn thiện dựa trên dữ liệu về một sản phẩm mà tôi cấp cho bạn:
        
        Dictionary: {content_dict}
        
        Hãy viết lại thành một câu hoàn thiện, phù hợp để dùng làm input cho mô hình embedding,không được tóm tắt,  tuyệt đối không được thiếu thông tin và không được bịa đặt thêm thông tin.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    return answer


def embedding_documents(documents, embeddings):
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    flattened_docs = [doc for doc in documents if doc.page_content]
    initial_chunks = text_splitter.split_documents(flattened_docs)

    semantic_chunker = SemanticChunker(embeddings=embeddings)
    chunks = semantic_chunker.split_documents(initial_chunks)

    return chunks


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
        

def save_to_db(chunks, connection_string, embeddings):
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

# embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=openai_api_key)


# text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# # flattened_docs = [Document(page_content=doc.page_content) for doc in documents if doc.page_content]
# flattened_docs = [doc for doc in documents if doc.page_content]
# initial_chunks = text_splitter.split_documents(flattened_docs)

# semantic_chunker = SemanticChunker(embeddings=embeddings)
# chunks = semantic_chunker.split_documents(initial_chunks)



# connection_string = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@localhost:5435/{postgres_db}"

# def create_connection():
#     try:
#         connection = psycopg2.connect(
#             user=postgres_user,
#             password=postgres_password,
#             host="localhost", 
#             port="5435",
#             database=postgres_db
#         )
#         return connection
#     except OperationalError as e:
#         print(f"The error '{e}' occurred")
#         return None

# connection = create_connection()

# if connection is not None:
#     PGVector.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         collection_name="products",
#         connection_string=connection_string,
#         pre_delete_collection=True,
#     )
# else:
#     print("Failed to connect to the database.")

if __name__ == "__main__":
    documents = get_document("/Users/ttcenter/AvocaEdu/demo_vector_store/data.json")
    content_dicts = get_content(documents)
    documents = []
    for content_dict in content_dicts:
        document = complete_sentense(content_dict)
        print(len(document))
        documents.append(Document(page_content=document))
    
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=openai_api_key)
    connection_string = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@localhost:5435/{postgres_db}"
    chunks = embedding_documents(documents, embeddings)
    save_to_db(chunks, connection_string, embeddings)