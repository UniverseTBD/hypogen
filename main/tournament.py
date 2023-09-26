# import yaml
# from langchain.chat_models import AzureChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from tqdm import tqdm
# import logging
# import json
# import time
# from concurrent.futures import ThreadPoolExecutor
# import concurrent.futures
# from novelty import main as calculate_novelty
# import os

# # Create ../logs/tournament directory if it doesn't exist
# log_dir = "../logs/tournament"
# os.makedirs(log_dir, exist_ok=True)

# # Configure logging
# log_file = os.path.join(log_dir, "tournament.log")
# logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')  # 'w' to overwrite the log file each run

# # This is an LLMChain to write a synopsis given a title of a play.
# config = yaml.safe_load(open("../config.yml"))
# API_KEY = config['api_key']
# DEPLOYMENT_NAME = config['deployment_name']
# BASE_URL = config['base_url']
# API_VERSION = config['api_version']

# llm = AzureChatOpenAI(
#     openai_api_base=BASE_URL,
#     openai_api_version=API_VERSION,
#     deployment_name=DEPLOYMENT_NAME,
#     openai_api_key=API_KEY,
#     openai_api_type="azure",
# )

# # New prompt template (modified to include novelty)
# tournament_template = """
# In this task, you are asked to utilise the Bit-Flip schema, a framework for evaluating scientific inquiries and their corresponding innovative solutions. The schema is based on two components:

# * **Bit**: Identifies the prevailing belief or 'status quo' that is being challenged.
# * **Flip**: Formulates a counterargument or innovative approach that contradicts or 'flips' the Bit.

# Given a Bit:

# {bit}

# You are tasked with comparing two Flips, designated as Flip A and Flip B:

# * **Flip A**: {flip_a} (Novelty Score: {novelty_a})
# * **Flip B**: {flip_b} (Novelty Score: {novelty_b})

# Please evaluate these Flips based on the following criteria, scoring each on a scale from 0 to 5:

# 1. **Creativity**: How creatively does the Flip challenge the Bit?
# 2. **Practicality**: How feasible is it to implement the Flip in real-world scenarios?

# After evaluating, please calculate the average score for each Flip, rounding to two decimal places, to determine which Flip is superior. Do this by averaging over the scores for novelty (given), creativity and practicality. 
# Your final answer should be the flip with the highest score. If it's Flip A, your last output should be 'Flip A.' If it's Flip B, your last output should be 'Flip B.'
# """

# # Initialise PromptTemplate
# tournament_prompt = PromptTemplate(
#     input_variables=["bit", "flip_a", "flip_b", "novelty_a", "novelty_b"],  # Include novelty_a and novelty_b
#     template=tournament_template
# )

# # Initialise LLMChain
# tournament_chain = LLMChain(llm=llm, prompt=tournament_prompt, output_key="judgement")

# # Knockout Tournament Function
# def run_knockout_tournament(bit, flips):
#     logging.info(f"Starting tournament for bit: {bit}")
#     logging.info(f"Initial Flips: {flips}")

#     remaining_flips = flips.copy()
#     novelty_scores = {}  # To store the novelty scores

#     # Calculate novelty scores for each flip
#     for flip in remaining_flips:
#         bit_flip = {'bit': bit, 'flip': flip}
#         novelty_scores[flip] = calculate_novelty(bit_flip)
    
#     while len(remaining_flips) > 1:
#         next_round_flips = []
#         logging.info(f"New Round: Remaining Flips: {remaining_flips}")

#         while len(remaining_flips) > 1:
#             next_round_flips = []
#             logging.info(f"New Round: Remaining Flips: {remaining_flips}")

#             # Pairwise comparison of remaining flips
#             with ThreadPoolExecutor() as executor:
#                 future_results = {executor.submit(tournament_chain, {
#                     "bit": bit,
#                     "flip_a": remaining_flips[i],
#                     "flip_b": remaining_flips[i+1],
#                     "novelty_a": novelty_scores[remaining_flips[i]],
#                     "novelty_b": novelty_scores[remaining_flips[i+1]]
#                 }): (remaining_flips[i], remaining_flips[i+1]) for i in range(0, len(remaining_flips), 2) if i + 1 < len(remaining_flips)}
                
#                 for future in concurrent.futures.as_completed(future_results):
#                     flip_a, flip_b = future_results[future]
#                     try:
#                         result = future.result()
#                     except Exception as e:
#                         logging.info(f"Exception while comparing {flip_a} and {flip_b}: {e}")
#                         continue
                    
#                     judgement = result["judgement"]
#                     winner = interpret_judgement(judgement, flip_a, flip_b)
#                     next_round_flips.append(winner)
            
#             remaining_flips = next_round_flips

#     logging.info(f"The winner of the tournament is: {remaining_flips[0]}")
#     return remaining_flips[0]  # The last remaining flip is the winner

# # Function to interpret judgement and return the winner
# def interpret_judgement(judgement, flip_a, flip_b):
#     last_a = judgement.rfind('A')
#     last_b = judgement.rfind('B')
    
#     if last_a > last_b: return flip_a
#     else: return flip_b

# if __name__ == "__main__":
#     # Example usage
    
#     # Load a bit 
#     path = "../data/generated/proposal_hypogen.json"
#     with open(path, 'r') as f:
#         proposal_data = json.load(f)

#     bit = list(proposal_data.keys())[0]
#     flips = proposal_data[bit]

#     # Run the knockout tournament and time it
#     start = time.time()
#     winner = run_knockout_tournament(bit, flips)
#     print(f"The winner is: {winner}")
#     end = time.time()
#     print(f"Time taken: {end - start:.2f} seconds")

import yaml
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import logging
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from novelty import main as calculate_novelty
import os

# Create ../logs/tournament directory if it doesn't exist
log_dir = "../logs/tournament"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, "tournament.log")
logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')  # 'w' to overwrite the log file each run

# Load config
config = yaml.safe_load(open("../config.yml"))
API_KEY = config['api_key']
DEPLOYMENT_NAME = config['deployment_name']
BASE_URL = config['base_url']
API_VERSION = config['api_version']

# Initialize LLM model
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
)

tournament_template = """
In this task, you are asked to utilise the Bit-Flip schema, a framework for evaluating scientific inquiries and their corresponding innovative solutions. The schema is based on two components:

* **Bit**: Identifies the prevailing belief or 'status quo' that is being challenged.
* **Flip**: Formulates a counterargument or innovative approach that contradicts or 'flips' the Bit.

Given a Bit:

{bit}

You are tasked with evaluating the following Flip:

* **Flip**: {flip} (Novelty Score: {novelty})

Please evaluate this Flip based on the following criteria, scoring each on a scale from 0 to 5:

1. **Creativity**: How creatively does the Flip challenge the Bit?
2. **Practicality**: How feasible is it to implement the Flip in real-world scenarios?

After evaluating, please calculate the average score for the Flip, rounding to two decimal places. Do this by averaging over the scores for novelty (given), creativity, and practicality.
Your final answer should be the average score.
"""

# Initialise PromptTemplate
tournament_prompt = PromptTemplate(
    input_variables=["bit", "flip", "novelty"],  # Include only bit, flip, and novelty
    template=tournament_template
)

# Initialize LLMChain
tournament_chain = LLMChain(llm=llm, prompt=tournament_prompt, output_key="judgement")

# Cache for previously computed scores
score_cache = {}

# Function to get the score for a given bit and flip
def get_flip_score(bit, flip, novelty):
    result = tournament_chain({
        "bit": bit,
        "flip": flip,
        "novelty": novelty
    })
    judgement_text = result["judgement"]
    
    # Use a regular expression to find the last floating-point number in the string
    last_float = re.findall(r"[-+]?\d*\.\d+|\d+", judgement_text)[-1]
    score = float(last_float)
    return score

# Function to interpret the score from the judgement
def interpret_score(judgement):
    # Implement your code here to interpret the score from the judgement
    # For demonstration, let's assume the judgement directly gives the score
    return float(judgement)

# Function to calculate the novelty score for a given flip
def calculate_novelty_score(bit, flip):
    return calculate_novelty({'bit': bit, 'flip': flip})

# Main function
def run_knockout_tournament(bit, flips):
    logging.info(f"Starting tournament for bit: {bit}")
    logging.info(f"Initial Flips: {flips}")

    # Parallelize the calculation of novelty scores
    with ThreadPoolExecutor() as executor:
        future_novelties = {executor.submit(calculate_novelty_score, bit, flip): flip for flip in flips}
        novelty_scores = {flip: future.result() for future in concurrent.futures.as_completed(future_novelties) for flip in [future_novelties[future]]}
    
    # Parallelize the scoring process
    with ThreadPoolExecutor() as executor:
        future_scores = {executor.submit(get_flip_score, bit, flip, novelty_scores[flip]): flip for flip in flips}
        
        # Wait for all futures to complete and collect the scores
        completed_scores = {flip: future.result() for future in concurrent.futures.as_completed(future_scores) for flip in [future_scores[future]]}
        
        # Find the flip with the maximum score
        best_flip = max(completed_scores, key=completed_scores.get)
        max_score = completed_scores[best_flip]
            
    logging.info(f"The winner of the tournament is: {best_flip} with a score of {max_score}")
    return best_flip

# Main execution
if __name__ == "__main__":
    path = "../data/generated/proposal_hypogen.json"
    with open(path, 'r') as f:
        proposal_data = json.load(f)

    bit = list(proposal_data.keys())[0]
    flips = proposal_data[bit]

    start = time.time()
    winner = run_knockout_tournament(bit, flips)
    end = time.time()

    print(f"The winner is: {winner}")
    print(f"Time taken: {end - start:.2f} seconds")
