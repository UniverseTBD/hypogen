import os
import yaml
from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

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

# Use JSONLoader to load each article from the JSON file
# We're using the jq_schema '.' to get each dictionary in the JSON array
loader = JSONLoader(
    file_path='../data/processed/extracted_preprocessed.json',
    jq_schema='.text'
)

documents = loader.load()
print(documents)

documents = loader.load()
print(documents)

# Each document will be a combination of the "Problem", "Solution", "Methodology", "Evaluation", and "Results" fields
documents = [doc['Problem'] + " " + doc['Solution'] + " " + doc['Methodology'] + " " + doc['Evaluation'] + " " + doc['Results'] for doc in documents]

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