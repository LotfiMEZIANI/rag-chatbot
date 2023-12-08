import os

from dotenv import load_dotenv

load_dotenv()

llm_model_name: str = os.environ.get("LLM_MODEL_NAME")
openai_api_key: str = os.environ.get("OPENAI_API_KEY")
embedding_model_name: str = os.environ.get("OPENAI_EMBEDDING_MODEL_NAME")
pinecone_api_key: str = os.environ.get("PINECONE_API_KEY")
pinecone_environment: str = os.environ.get("PINECONE_ENVIRONMENT")
pinecone_index_name: str = os.environ.get("PINECONE_INDEX_NAME")
sources_folder_path: str = os.environ.get("SOURCES_FOLDER_PATH")
vector_text_field: str = os.environ.get("VECTOR_TEXT_FIELD")
