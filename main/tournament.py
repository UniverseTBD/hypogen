import yaml
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import logging
import os

# Create ../logs/tournament directory if it doesn't exist
log_dir = "../logs/tournament"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, "tournament.log")
logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')  # 'w' to overwrite the log file each run

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

# New Prompt Template
tournament_template = """
In this task, you are asked to utilise the Bit-Flip schema, a framework for evaluating scientific inquiries and their corresponding innovative solutions. The schema is based on two components:

* **Bit**: Identifies the prevailing belief or 'status quo' that is being challenged.
* **Flip**: Formulates a counterargument or innovative approach that contradicts or 'flips' the Bit.

Given a Bit:

{bit}

You are tasked with comparing two Flips, designated as Flip A and Flip B:

* **Flip A**: {flip_a}
* **Flip B**: {flip_b}

Please evaluate these Flips based on the following criteria, scoring each on a scale from 0 to 5:

1. **Novelty**: Does the Flip introduce a new idea or perspective?
2. **Creativity**: How creatively does the Flip challenge the Bit?
3. **Efficiency**: Is the Flip likely to be more efficient in achieving the intended outcome than the Bit?
4. **Practicality**: How feasible is it to implement the Flip in real-world scenarios?
5. **Elegance**: Does the Flip provide a more elegant or simpler solution than the Bit?

After evaluating, please calculate the average score for each Flip, rounding to two decimal places, to determine which Flip is superior.
Your final answer should be the flip with the highest score. If it's Flip A, your last output should be 'Flip A.' If it's Flip B, your last output should be 'Flip B.'
"""

# Initialise PromptTemplate
tournament_prompt = PromptTemplate(input_variables=["bit", "flip_a", "flip_b"], template=tournament_template)

# Initialise LLMChain
tournament_chain = LLMChain(llm=llm, prompt=tournament_prompt, output_key="judgement")

# Knockout Tournament Function
def run_knockout_tournament(bit, flips):
    remaining_flips = flips.copy()
    logging.info(f"Starting tournament for bit: {bit}")
    logging.info(f"Initial Flips: {flips}")
    
    while len(remaining_flips) > 1:
        next_round_flips = []
        logging.info(f"New Round: Remaining Flips: {remaining_flips}")
        
        # Pairwise comparison of remaining flips
        for i in range(0, len(remaining_flips), 2):
            if i + 1 < len(remaining_flips):  # Ensure there's a pair
                flip_a = remaining_flips[i]
                flip_b = remaining_flips[i+1]
                result = tournament_chain({"bit": bit, "flip_a": flip_a, "flip_b": flip_b})
                judgement = result["judgement"]
                
                # Interpret judgement to choose winner
                winner = interpret_judgement(judgement, flip_a, flip_b)
                logging.info(f"Match between Flip A: {flip_a} and Flip B: {flip_b}. Judgement: {judgement}. Winner: {winner}")
                
                next_round_flips.append(winner)
            else:  # If an odd number, the last one automatically moves to the next round
                logging.info(f"Unmatched Flip: {remaining_flips[i]} moves to the next round.")
                next_round_flips.append(remaining_flips[i])
                
        remaining_flips = next_round_flips

    logging.info(f"The winner of the tournament is: {remaining_flips[0]}")
    return remaining_flips[0]  # The last remaining flip is the winner

# Function to interpret judgement and return the winner
def interpret_judgement(judgement, flip_a, flip_b):
    last_a = judgement.rfind('A')
    last_b = judgement.rfind('B')
    
    if last_a > last_b: return flip_a
    else: return flip_b

# Example usage
bit = "Probabilistic principal component analysis (PPCA) seeks a low dimensional representation of a data set by solving an eigenvalue problem on the sample covariance matrix, assuming independent spherical Gaussian noise."

flips = ["Rather than solely utilizing PPCA for low-dimensional representation, employ a Multi-Layer Perceptron (MLP) as an autoencoder. This approach allows for the capturing of non-linear relationships in the data, enhancing the feature extraction process.", 
         "Instead of PPCA, use Canonical Correlation Analysis (CCA) to find the directions that maximize the correlation between variables. This can be especially useful when the dataset involves multiple types of data, allowing for a richer representation.", 
         "Forego PPCA in favor of t-Distributed Stochastic Neighbor Embedding (t-SNE) to find a low-dimensional representation. t-SNE excels in preserving local structure and can be particularly useful when clusters in high-dimensional space need to be accurately reflected in the low-dimensional mapping.", 
         "Instead of solely relying on PPCA, the data variance can be further decomposed into its components through a generalised eigenvalue problem, called residual component analysis (RCA), which considers other factors like sparse conditional dependencies and temporal correlations that leave some residual variance."]
winner = run_knockout_tournament(bit, flips)
print(f"The winner is: {winner}")