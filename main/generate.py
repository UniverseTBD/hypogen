import os
import re
import time
import yaml
import pandas as pd
from tqdm import tqdm
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, validator

category = 'cs.LG'
METHOD = "gpt_4_three_shot"
INPUT_DIR = "../data/tuning"
OUTPUT_DIR = "../data/generated"
INPUT_FILE_NAME = f"{category}_test.csv"  # Assuming "_test" is added to the test set
OUTPUT_FILE_NAME = f"{METHOD}.csv"

ZERO_SHOT_PROMPT = """
Consider the Bit-Flip concept:

* Bit: The prevailing belief or conventional approach in a given domain.
* Flip: The counterargument or innovative approach that challenges or overturns the 'Bit'.

Bit-Flip Defined:
A Bit-Flip inverts a commonly held assumption, often questioning existing constraints, reapplying techniques to new domains, or adapting solutions for different scales. The 'Bit' is the prevailing belief, and the 'Flip' is the counterargument that challenges it.

Your task is to generate the 'Flip' based on the provided 'Bit' in the field of machine learning:

* Bit: {bit}

Please articulate a 'Flip' that logically counters or innovates upon the given 'Bit'. Your 'Flip' should consist of three sentences that are logically connected, providing a holistic view of the innovative approach or counterargument.

Remember, avoid phrases like "This research aims to..." or "This paper proposes...", and instead focus on describing the innovative approach or counterargument directly. Your flip should be creative, novel, practical and elegant.

Do not provide anything else except your three sentence flip. Provide your flip (the innovative counterargument or approach). Flip: 
"""

THREE_SHOT_PROMPT = """
Consider the Bit-Flip concept:

* Bit: The prevailing belief or conventional approach in a given domain.
* Flip: The counterargument or innovative approach that challenges or overturns the 'Bit'.

Your task is to generate the 'Flip' based on the provided 'Bit' in the field of machine learning. Please articulate a 'Flip' that logically counters or innovates upon the given 'Bit'. Your 'Flip' should consist of three sentences that are logically connected, providing a holistic view of the innovative approach or counterargument.

### BIT: Traditional statistical learning constructs a predictor of a random variable Y as a function of a related random variable X, based on a training sample from their joint distribution. The goal is to approach the performance of the best predictor in the specified class. This approach assumes perfect observation of the X-part of the sample, while the Y-part is communicated at some finite bit rate.
### FLIP: The research proposes a setting where the encoding of the Y-values is allowed to depend on the X-values. Under certain conditions on the admissible predictors, the underlying family of probability distributions, and the loss function, an information-theoretic characterization of achievable predictor performance is given. This is based on conditional distortion-rate functions, illustrating a new approach to nonparametric regression in Gaussian noise.
### END

### BIT: In sensor networks, the communication among sensors is often subject to random errors, costs, and constraints due to limited resources. The signal-to-noise ratio (SNR) is typically a key factor in determining the probability of communication failure in a link. The conventional approach to this problem has been to accept these probabilities as a proxy for the SNR under which the links operate.
### FLIP: This research proposes a new approach to designing the topology of sensor networks, taking into account the probabilities of reliable communication among sensors and link failures. The study models the network as a random topology and establishes conditions for convergence of average consensus when network links fail. By formulating topology design as a constrained convex optimization problem, the research shows that the optimal design significantly improves the convergence speed of the consensus algorithm and can achieve the performance of a non-random network at a fraction of the communication cost.
### END

### BIT: Traditional algorithms for the online shortest path problem operate under the assumption that the decision maker has complete knowledge of the edge weights in a graph. This approach is based on the belief that the decision maker can make the most optimal choice when all information is available. However, this does not account for scenarios where the edge weights can change arbitrarily, and the decision maker only learns the weights of the edges that belong to the chosen path.
### FLIP: An innovative approach is to develop an algorithm that can handle partial monitoring, where the decision maker only learns the weights of the edges that belong to the chosen path. This algorithm's average cumulative loss over n rounds only exceeds that of the best path, matched offline to the entire sequence of edge weights, by a quantity proportional to 1/√n. This approach can be extended to other settings, such as label efficient setting and tracking the best expert, and can be implemented with linear complexity in the number of rounds and edges.
### END

### BIT: {bit}
### FLIP:
"""

class FlipExtractionPromptTemplate(StringPromptTemplate, BaseModel):
    @validator("input_variables")
    def validate_input_variables(cls, v):
        if len(v) != 1 or "bit" not in v:
            raise ValueError("bit must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        prompt = THREE_SHOT_PROMPT.format(bit=kwargs["bit"])
        return prompt

    def _prompt_type(self):
        return "flip-extraction"

class Flip(BaseModel):
    Flip: str = Field(description="The innovative perspective or approach.")

config = yaml.safe_load(open("../config.yml"))
API_KEY = config['api_key']
DEPLOYMENT_NAME = config['deployment_name']
BASE_URL = config['base_url']
API_VERSION = config['api_version']

model = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0.1,
    max_tokens=300,
)

def generate_custom_prompt(bit):
    return THREE_SHOT_PROMPT.format(bit=bit)

def get_model_response(bit):
    prompt = generate_custom_prompt(bit)
    messages = [HumanMessage(content=prompt)]
    response = model(messages)
    return response.content

def extract_to_flip(response: str) -> str:
    # Directly return the model's response as a string
    result = response.strip()
    # Remove the "Flip: " from the start if it exists
    if result.startswith("Flip: "):
        result = result[len("Flip: "):]
    return result

def worker(bit):
    try:
        response = get_model_response(bit)
        flip = extract_to_flip(response)
        time.sleep(5)
        return {'bit': bit, 'flip': flip}  # Return as a dictionary
    except Exception as e:
        print(f"Error processing bit: {bit[:100]}... Error: {e}")
        time.sleep(5)
        return None

def main():

    # Read bits from the test set CSV
    input_file_path = os.path.join(INPUT_DIR, INPUT_FILE_NAME)
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file {input_file_path} does not exist. Please check the path and try again.")
    
    df = pd.read_csv(input_file_path)
    bits = df['bit'].dropna().tolist()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    existing_data = []
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        existing_data = df_existing.to_dict('records')

    with ThreadPoolExecutor(max_workers=5) as executor:
        for result in tqdm(executor.map(worker, bits), total=len(bits), desc="Generating Flips"):
            if result:
                existing_data.append({'method': METHOD, **result})  # Add the result dictionary directly

    df_new = pd.DataFrame(existing_data)
    df_new.to_csv(file_path, index=False)

if __name__ == "__main__":
    main()
