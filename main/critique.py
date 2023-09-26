import ast
import yaml
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA  
from preprocessing import Hypothesis
from tqdm import tqdm
import logging
import os
from expand import expand_flip

# Create logging directory if it doesn't exist
log_dir = "../logs/critique"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=f'{log_dir}/critique.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This is an LLMChain to write a synopsis given a title of a play.
config = yaml.safe_load(open("../config.yml"))
API_KEY = config['api_key']
DEPLOYMENT_NAME = config['deployment_name']
BASE_URL = config['base_url']
API_VERSION = config['api_version']

llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)

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


vectordb = FAISS.load_local("../data/vectorstore/arxiv-cs.LG", embeddings)

# Critic Chain: Takes a hypothesis and provides a critique.
critic_template = """
Given the following hypothesis:
{hypothesis}

And the following relevant papers:
{context}

Analyse each part of the hypothesis:
1. Clarity: Are the statements clear and easy to understand?
2. Coherence: Do the different parts of the hypothesis logically flow together? Does it seem like the proposed solution would work?
3. Scientific Validity: Are there any scientific inaccuracies or assumptions that seem unfounded?

After your analysis, provide specific feedback on how the flip can be improved, either conceptually (changing the idea) or microscopically (changing the implementation). 
Specifically, give feedback on the frailties of the idea as a whole, and suggest potential enhancements.
"""

reviser_template = """
You are tasked with revising an original hypothesis based on a critique. Here are the details:

Original Hypothesis: {hypothesis}
Critique: {critique}

Your revised hypothesis should:
- Be limited to three sentences, but don't number them. Don't make the sentences overly long.
- Address each point in the critique with a specific solution in the hypothesis. Don't just say "will be further explored" or acknowledge problems without providing solutions.
- Be scientifically valid, clear, and coherent.
- Maximize information density and conciseness of the three-sentence hypothesis.
- Prioritize scientific specifics and insight over generalities. Do not be trite or vague.
- Be as feasible, accurate, and creative as the critique allows.

Please only write the revised hypothesis.
"""

def retrieve_related_papers(hypothesis, vectordb, top_k=5):
    query = hypothesis
    top_documents = vectordb.similarity_search(query, top_k=top_k)
    return [doc.page_content for doc in top_documents]


def adversarial_update_hypothesis(hypothesis):
    # Retrieve top 5 related papers
    related_papers = retrieve_related_papers(hypothesis, vectordb)
    # Format related papers into a string
    context_str = "\n".join([f"- {paper}" for paper in related_papers])
    # Prepare the input for the critic chain
    critic_input = {
        "hypothesis": hypothesis,
        "context": context_str  # Add the context here
    }
    critic_prompt = PromptTemplate(input_variables=["hypothesis", "context"], template=critic_template)  # Add 'context' here
    critic_chain = LLMChain(llm=llm, prompt=critic_prompt, output_key="critique")
    reviser_prompt = PromptTemplate(input_variables=["hypothesis", "critique"], template=reviser_template)
    reviser_chain = LLMChain(llm=llm, prompt=reviser_prompt, output_key="revised_hypothesis")
    overall_chain = SequentialChain(
        chains=[critic_chain, reviser_chain],
        input_variables=["hypothesis", "context"],  # Add 'context' here
        output_variables=["critique", "revised_hypothesis"],
        verbose=False)
    result = overall_chain({"hypothesis": str(hypothesis), "context": context_str})
    return result

def improve_hypothesis(hypothesis: dict, n_iters: int = 3) -> dict:
    expanded_flip = expand_flip(hypothesis['Flip'])
    logging.info(f"Expanded Flip: {expanded_flip}")
    for i in tqdm(range(n_iters)):
        revised_hypothesis = adversarial_update_hypothesis(expanded_flip)
        expanded_flip = revised_hypothesis['revised_hypothesis']
        logging.info(f"Iteration {i+1}: Revised hypothesis is: {expanded_flip}")

    # Returning a dictionary containing "Bit", "Flip", and "Final" hypothesis
    return {
        'Bit': hypothesis['Bit'],
        'Flip': hypothesis['Flip'],
        'Final': expanded_flip
    }

if __name__ == "__main__":

    hypothesis_to_improve = {
        'Bit': 'Statistical model estimation in sensor networks requires advanced and costly joint optimization methods for distributed learning.',
        'Flip': 'Simple linear combination or max-voting methods, when combined with second-order information, can be statistically competitive, offering low communication and computational cost and "any-time" behavior.'
    }

    improved_hypothesis = improve_hypothesis(hypothesis=hypothesis_to_improve, n_iters=3)
    logging.info(f"Final improved hypothesis: {improved_hypothesis}")
    print(improved_hypothesis)