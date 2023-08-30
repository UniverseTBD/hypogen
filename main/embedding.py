from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import yaml
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

def worker(text):
    """Worker function to compute embeddings for a text."""
    if text is None:
        return None
    embedded_text = embeddings.embed_query(text)
    return (text, embedded_text)

# Load configuration and set API key
config = yaml.safe_load(open("../config.yml"))
API_KEY = config['embedding_api_key']
DEPLOYMENT_NAME = "embedding"
BASE_URL = config['embedding_base_url']
os.environ['OPENAI_API_KEY'] = API_KEY

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    openai_api_base=BASE_URL,
    deployment=DEPLOYMENT_NAME,
    model="text-embedding-ada-002",
    openai_api_key=API_KEY,
    openai_api_type="azure",
    chunk_size = 16,
)

# Load CSV file directly
df = pd.read_csv('../data/processed/arxiv-cs.LG.csv', low_memory=False)

# Create a list of strings by concatenating 'title' and 'abstract'
documents = [f"{row['title']} {row['abstract']}" for index, row in df.head(3000).iterrows()]

# Initialize an empty list to hold (text, embedding) tuples
text_embeddings = []

# Use ThreadPoolExecutor with max_workers for threading
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(tqdm(executor.map(worker, documents), total=len(documents), desc="Computing Embeddings"))
    text_embeddings.extend([(text, embedding) for text, embedding in results if text and embedding])

# Create a FAISS Vectorstore and add the embeddings
vectordb = FAISS.from_embeddings(text_embeddings, embedding=embeddings)
vectordb.save_local("../data/vectorstore/arxiv-cs.LG")
