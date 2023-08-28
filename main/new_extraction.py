from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator, Field
import yaml
import re
from langchain.chat_models import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from tqdm import tqdm
from typing import Dict
import concurrent.futures
from utils import *
import os

PROMPT = """\
Please conduct a structured analysis of the following academic paper:

{abstract}

Your analysis should clearly and succinctly capture the paper's problem and proposed solution. Ensure that the bullet points in the "Solution" section are logically connected and follow the sequence proposed in the paper.

Adhere to the following specific format:
* Problem: Summarize the main problem or research question the paper is trying to address in a single, concise sentence. Frame this as a general question in science, not in the context of the paper i.e. "LSTMs and RNNs in general cannot be parallelised and thus are slow" vs "This paper aims to address the lack of parallelisability of RNNs".
* Solution: List 4-6 bullet points outlining the key steps or methods the paper proposes to solve the identified problem or answer the research question. Each bullet point should encapsulate a distinct step or method.

Your analysis should clearly and succinctly capture the paper's problem and proposed solution. Ensure that the bullet points in the "Solution" section are logically connected and follow the sequence proposed in the paper.
"""

class HypothesisExtractionPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template for extracting a structured representation from an abstract."""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "abstract" not in v:
            raise ValueError("abstract must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        # Generate the prompt using the provided abstract
        prompt = PROMPT.format(abstract=kwargs["abstract"])
        return prompt

    def _prompt_type(self):
        return "hypothesis-extraction"

config = yaml.safe_load(open("../config.yml"))
API_KEY = config['api_key']
DEPLOYMENT_NAME = config['deployment_name']
BASE_URL = config['base_url']
API_VERSION = config['api_version']

# 1. Setup the chat model
model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0.1,
)

# 2. Define the custom prompt
def generate_custom_prompt(abstract):
    return PROMPT.format(abstract=abstract)

# 3. Get model's response
def get_model_response(abstract):
    prompt = generate_custom_prompt(abstract)
    messages = [HumanMessage(content=prompt)]
    response = model(messages)
    return response.content


def worker(abstract):
    """Function to be executed by each thread."""
    try:
        hypothesis = get_model_response(abstract)
        return hypothesis
    except Exception as e:
        print(f"Error processing abstract: {abstract[:100]}... Error: {e}")
        return None


if __name__ == "__main__":
    df = pd.read_csv('../data/raw/abstract.csv', low_memory=False)
    
    abstracts = df['Abstract'].values
    
    # List to store the extracted hypotheses
    extracted_hypotheses = []

    # Use ThreadPoolExecutor to execute multiple abstracts concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor: # For example, using 5 threads
        # Process each abstract
        for result in tqdm(executor.map(worker, abstracts), total=len(abstracts), desc="Extracting Hypotheses"):
            if result:
                extracted_hypotheses.append(result)

    # Save the extracted hypotheses to a CSV file
    df = pd.DataFrame(extracted_hypotheses)
    df.to_csv('../data/processed/new_extracted_hypotheses.csv', index=False)