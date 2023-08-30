import os
import yaml
import re
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

def parse_novelty_score(output):
    match = re.search(r'Novelty Score: (\d)', output)
    if match:
        return int(match.group(1))
    else:
        return "Could not extract novelty score."

def main(bit_flip):
    # Load configuration and set API key
    config = yaml.safe_load(open("../config.yml"))
    API_KEY = config['embedding_api_key']
    DEPLOYMENT_NAME = "embedding"
    BASE_URL = config['embedding_base_url']
    API_VERSION = config['api_version']
    os.environ['OPENAI_API_KEY'] = API_KEY

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        openai_api_base=BASE_URL,
        deployment=DEPLOYMENT_NAME,
        model="text-embedding-ada-002",
        openai_api_key=API_KEY,
        openai_api_type="azure",
        chunk_size=16,
    )

    # Initialize chat model
    llm = AzureChatOpenAI(
        openai_api_base=config['base_url'],
        openai_api_version=API_VERSION,
        deployment_name=config['deployment_name'],
        openai_api_key=config['api_key'],
        openai_api_type="azure",
        temperature=0.0,
    )

    # Load Documents
    loader = CSVLoader(file_path='../data/processed/arxiv-cs.LG.csv', source_column="authors")
    documents = loader.load()
    documents = documents[:50]

    # Create Chroma Vectorstore
    vectordb = FAISS.load_local("../data/vectorstore/arxiv-cs.LG", embeddings)

    prompt_template = """
    You are a Neurips paper review determining the novelty of an idea. Use the following pieces of context to determine if the following idea.

    Please perform a concise analysis to evaluate the novelty of this "flip" by:

    * Describing the originality of the new idea in relation to the original "bit".
    * Comparing it to existing ideas or implementations that are similar.
    * Assessing its potential impact or utility in its domain.
    * Summing up the above findings.

    Here are some other relevant ideas from paper abstracts that you can use for assessing novelty:

    {context}

    Here is the bit-flip: {question}

    Based on this context and your own knowledge, determine the novelty of the flip component of the idea. Please provide your final output as a novelty score between 0 to 5, where 0 means "Not Novel" and 5 means "Highly Novel". 
    If the idea already exists in the provided abstracts, provide a low score. If the idea is incremental or not very different, also provide a low score.
    Your final output should be of the form 'Novelty Score: x'.
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(), chain_type_kwargs=chain_type_kwargs)

    query = f"Bit: {bit_flip['bit']}. Flip: {bit_flip['flip']}"
    result = qa.run(query)
    score = parse_novelty_score(result)
    return score

if __name__ == "__main__":
    bit_flip = {
        'bit': """Probabilistic principal component analysis (PPCA) seeks a low dimensional representation of a data set by solving an eigenvalue problem on the sample covariance matrix, assuming independent spherical Gaussian noise.""",
        'flip': """Instead of solely relying on PPCA, the data variance can be further decomposed into its components through a generalised eigenvalue problem, called residual component analysis (RCA), which considers other factors like sparse conditional dependencies and temporal correlations that leave some residual variance.""",
    }

    novelty_score = main(bit_flip)
    print(f"Novelty Score: {novelty_score}")