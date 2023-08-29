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

if __name__ == "__main__":

    yuan = True

    # Load the data
    if yuan: 
        df = pd.read_csv('../data/raw/yuan.csv', low_memory=False).iloc[:3]
    else: 
        df = pd.read_csv('../data/raw/arxiv.csv', low_memory=False)

    # Shuffle and limit the number of abstracts
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    abstracts = df['abstract'].values
    print(f"Extracting hypotheses from {len(abstracts)} abstracts...")
    
    extracted_bits = [None] * len(abstracts)  # Initialize with None to preserve order
    extracted_flips = [None] * len(abstracts)  # Initialize with None to preserve order
    titles = [None] * len(abstracts)  # Initialize with None to preserve order

    # Multithreaded processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, result in tqdm(executor.map(worker, enumerate(abstracts)), total=len(abstracts), desc="Extracting Hypotheses"):
            extracted_bits[index] = result['Bit']
            extracted_flips[index] = result['Flip']
            titles[index] = df['title'].values[index]

    # Create DataFrame from extracted hypotheses and titles
    df_extracted = pd.DataFrame({
        'bit': extracted_bits,
        'flip': extracted_flips,
        'title': titles
    }).dropna()

    # Train-test split: 90-10
    df_train = df_extracted.iloc[int(0.1*len(df_extracted)):]
    df_test = df_extracted.iloc[:int(0.1*len(df_extracted))]

    # Save dataframe to CSV
    if yuan:
        df_train.to_csv('../data/processed/yuan_train.csv', index=False)
        df_test.to_csv('../data/processed/yuan_test.csv', index=False)
    else:
        df_train.to_csv('../data/processed/train.csv', index=False)
        df_test.to_csv('../data/processed/test.csv', index=False)