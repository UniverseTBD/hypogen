import os
import yaml
import json
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader

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
    chunk_size=16,
)

# Load the JSON file and extract the 'text' attribute for each dictionary
csv_file_path = '../data/processed/arxiv-cs.LG.csv'
loader = CSVLoader(file_path=csv_file_path, 
                   #csv_args={"fieldnames": ["title", "abstract"]})#,
                   source_column="authors")
documents = loader.load()
print(documents[0])

# Define the persistence directory for Chroma
persist_directory = "chroma_db"

# Create the persistent Chroma DB instance
vectordb = Chroma.from_documents(
    documents=documents, embedding=embeddings, persist_directory=persist_directory
)

# Persist the embeddings
vectordb.persist()

# # Load from disk and query it
# query = "special metallicity"
# db3 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# docs = db3.similarity_search(query)
# # print(docs[0])