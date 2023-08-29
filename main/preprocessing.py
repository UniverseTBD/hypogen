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
from concurrent.futures import ThreadPoolExecutor
from utils import *
import os

PROMPT = """
Before you begin, let's understand the Bit-Flip concept using the example of BERT in NLP:

Example:

* Bit: NLP models read sentences word by word to understand the preceding context.
* Flip: NLP models should read the entire sentence at once to understand both the preceding and succeeding context.

Bit-Flip Defined:
A Bit-Flip inverts a commonly held assumption, often questioning existing constraints, reapplying techniques to new domains, or adapting solutions for different scales. The 'Bit' is the prevailing belief, and the 'Flip' is the counterargument that challenges it.

Guidance: When capturing the Bit and Flip, state them directly without referring to the specific research. Avoid phrases like "This paper aims to..." or "This research challenges...".

Now, consider the following research abstract:

{abstract}

Your task is to articulate the abstract using a Bit-Flip schema:

* Bit: Identify the conventional belief or 'status quo' the abstract implicitly or explicitly challenges.
* Flip: Formulate the counterargument or innovative approach that flips the 'Bit'.

Capture these insights in a JSON-style dictionary:

{{
'Bit': 'Conventional belief or assumption being challenged.',
'Flip': 'Innovative counterargument or approach.'
}}

Your goal is to encapsulate the essence of the research abstract using only the Bit and Flip, while adhering to the Bit-Flip schema.
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

# Redefining the Pydantic model for Hypothesis
class Hypothesis(BaseModel):
    Bit: str = Field(description="Conventional belief or approach that is being challenged.")
    Flip: str = Field(description="The innovative perspective that sets this research apart.")

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

# 4. Parse the response to get the structured representation
def extract_to_hypothesis(response: str) -> Hypothesis:
    # Extract content between curly braces
    match = re.search(r'\{(.+?)\}', response, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find a JSON-like structure in the response.")
    content = match.group(1)
    # Split by lines and extract key-value pairs
    lines = content.strip()
    # Remove blank lines
    lines = re.sub(r'\n\s*\n', '\n', lines)
    lines = lines.split('\n')
    data: Dict[str, str] = {}
    for line in lines:
        key, value = line.split(':', 1) # Split by the first colon
        key = key.strip().strip("'")
        value = value.strip().strip("',")
        data[key] = value
    # Populate the Hypothesis class
    return Hypothesis(**data)

# 4. Parse the response to get the structured representation
def extract_hypothesis(abstract):
    response = get_model_response(abstract)
    try:
        return extract_to_hypothesis(response)
    except Exception as e:
        print(f"Error extracting hypothesis from abstract with response: {response}... Error: {e}")

def worker(abstract_index_tuple):
    """Function to be executed by each thread."""
    index, abstract = abstract_index_tuple
    try:
        hypothesis = extract_hypothesis(abstract)
        return index, hypothesis.dict()
    except Exception as e:
        print(f"Error processing abstract: {abstract[:100]}... Error: {e}")
        return index, None

def main(n):
    # Set variables
    yuan = False
    category = 'cs.LG'

    # Read the last processed index from the existing CSV
    file_path = f'../data/tuning/{category}.csv' if not yuan else '../data/tuning/yuan.csv'
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        start_index = df_existing['index'].max() + 1  # Start from the next index
    else:
        start_index = 0  # Start from the beginning if the file doesn't exist

    # Load the data
    if yuan: 
        df = pd.read_csv('../data/processed/yuan.csv', low_memory=False).iloc[start_index:start_index+n]
    else: 
        df = pd.read_csv(f'../data/processed/arxiv-{category}.csv', low_memory=False).iloc[start_index:start_index+n]

    df = df.reset_index(drop=True)
    num_abstracts = len(df)
    
    print(f"Extracting hypotheses from {num_abstracts} abstracts starting from index {start_index}...")

    abstracts = df['abstract'].values
    extracted_bits = [None] * num_abstracts
    extracted_flips = [None] * num_abstracts
    titles = [None] * num_abstracts
    indices = [start_index + i for i in range(num_abstracts)]

    # Multithreaded processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, result in tqdm(executor.map(worker, enumerate(abstracts)), total=num_abstracts, desc="Extracting Hypotheses"):
            extracted_bits[index] = result['Bit'] if result else None
            extracted_flips[index] = result['Flip'] if result else None
            titles[index] = df['title'].values[index]

    # Create DataFrame
    df_extracted = pd.DataFrame({
        'index': indices,
        'bit': extracted_bits,
        'flip': extracted_flips,
        'title': titles
    }).dropna()

    # Append to existing CSV or create a new one
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_extracted = pd.concat([df_existing, df_extracted], ignore_index=True)

    df_extracted.to_csv(file_path, index=False)

    print(f"Finished processing up to index {start_index + num_abstracts - 1}")

if __name__ == "__main__":
    n = 10  # Number of abstracts to process in each run
    main(n)