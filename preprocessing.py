from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator, Field
import openai
import yaml
import re
from langchain.chat_models import AzureChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tqdm import tqdm
from typing import Dict
from utils import *
import os

PROMPT = """\
Please consider the following abstract:

{abstract}

I'd like to extract a structured representation from it. To achieve this, do the following steps by thinking out loud, and thinking step by step:

Begin by performing syntactic simplification of complex sentences in the abstract. 
Break them down into simpler, more digestible statements, but keep the key details intact, particularly the context, field of study, and application of the problem.
Next, organise the main concepts into a JSON-style object with the following structure:

- 'Problem': 'Issues or challenges addressed.',
- 'Solution': 'Proposed solutions or methods.'
- 'Methodology': 'Implementation details of the solutions or methods.'
- 'Evaluation': 'How results or solutions are assessed.'
- 'Results': 'Conclusions drawn from the study.'

Your goal is to capture the essence of the abstract in a clear and structured manner, highlighting the most critical elements using the provided categories.
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
    Problem: str = Field(description="Issues or challenges addressed.")
    Solution: str = Field(description="Proposed solutions or methods.")
    Methodology: str = Field(description="Implementation details of the solutions or methods.")
    Evaluation: str = Field(description="How results or solutions are assessed.")
    Results: str = Field(description="Conclusions drawn from the study.")

config = yaml.safe_load(open("config.yml"))
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
    lines = content.strip().split('\n')
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
    return extract_to_hypothesis(response)

if __name__ == "__main__":
    if not os.path.exists('arxiv.csv'):  df = load_arxiv_json()
    else:  df = pd.read_csv('arxiv.csv')
    
    abstracts = df['abstract'].values
    
    # List to store the extracted hypotheses
    extracted_hypotheses = []

    # Process each abstract
    for abstract in tqdm(abstracts, desc="Extracting Hypotheses"):
        try:
            hypothesis = extract_hypothesis(abstract)
            extracted_hypotheses.append(hypothesis.dict())
        except Exception as e:
            print(f"Error processing abstract: {abstract[:100]}... Error: {e}")

    # Save the extracted hypotheses to a JSON file
    with open("extracted.json", "w") as f:
        json.dump(extracted_hypotheses, f, indent=4)