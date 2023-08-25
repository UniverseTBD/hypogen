import os
import yaml
import json
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Load configuration and set API key
config = yaml.safe_load(open("../config.yml"))
API_KEY = config['embedding_api_key']
DEPLOYMENT_NAME = "embedding"
BASE_URL = config['embedding_base_url']
API_VERSION = config['api_version']
os.environ['OPENAI_API_KEY'] = API_KEY

embeddings = OpenAIEmbeddings(
    openai_api_base=BASE_URL,
    deployment=DEPLOYMENT_NAME,
    model="text-embedding-ada-002",
    openai_api_key=API_KEY,
    openai_api_type="azure",
)

# Load the JSON file and extract the 'text' attribute for each dictionary
file_path = '../data/processed/extracted_preprocessed.json'
json_data = json.loads(Path(file_path).read_text())
documents = [Document(page_content=entry['text'], metadata={"source": "local"}) for entry in json_data]
print(documents)

# Define the persistence directory for Chroma
persist_directory = "chroma_db"

# Create the persistent Chroma DB instance
vectordb = Chroma.from_documents(
    documents=documents, embedding=embeddings, persist_directory=persist_directory
)

# Persist the embeddings
vectordb.persist()

# Load from disk and query it
query = "Black hole"
db3 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
docs = db3.similarity_search(query)
print(docs[0].page_content)