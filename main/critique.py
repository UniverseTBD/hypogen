import ast
import yaml
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re
from preprocessing import Hypothesis
from tqdm import tqdm
import logging
import os

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

# Expansion Template: Expands the Flip into 4-6 bullet points
expansion_template = """
In this task, you are working within the framework of the Bit-Flip schema. The Bit-Flip schema is employed to rethink existing paradigms ('Bits') in scientific research by proposing innovative counterarguments or approaches ('Flips'). You've already been provided with a Flip that challenges a conventional Bit:

**Given Flip**: {flip}

Your mission is to expand upon this Flip by outlining a logical sequence of key steps or methods that could be employed to implement or test it. Consider this as sketching a roadmap for a research paper or project that seeks to validate the given Flip.

Please adhere to the following guidelines:
- List 4-5 bullet points.
- Each bullet point should encapsulate a distinct, logically connected step or method.
- Maintain alignment with the given Flip, ensuring each step contributes to its implementation or validation.
- Write your bullet points as complete sentences.

Here's an example for clarity:

**Given Flip**: NLP models should read the entire sentence at once to understand both the preceding and succeeding context.

- Design a Transformer-based neural network architecture with multi-headed attention mechanisms capable of ingesting entire sentences, optimizing for both computational efficiency and contextual understanding.
- Implement dynamic tokenization and embedding techniques that use bi-directional context to generate sentence-level embeddings, integrating this into the Transformer architecture.
- Utilize techniques like gradient clipping and batch normalization to stabilize the training process, employing large-scale datasets to train the model.
- Conduct a comprehensive evaluation using benchmarks like GLUE and SQuAD, comparing the model's performance against traditional word-by-word processing NLP models to quantify improvements in various tasks such as sentiment analysis, question-answering, and machine translation.

Remember, your bullet points should serve as a coherent and logical plan for implementing or validating the given Flip. 
"""

def expand_flip(flip: str) -> str:
    expansion_prompt = PromptTemplate(input_variables=["flip"], template=expansion_template)
    expansion_chain = LLMChain(llm=llm, prompt=expansion_prompt, output_key="expanded_flip")
    result = expansion_chain({"flip": flip})
    return result['expanded_flip']

# Critic Chain: Takes a hypothesis and provides a critique.
critic_template = """
Given the following hypothesis:
{hypothesis}

Analyse each part of the hypothesis:
1. Clarity: Are the statements clear and easy to understand?
2. Coherence: Do the different parts of the hypothesis logically flow together? Does it seem like the proposed solution would work?
3. Scientific Validity: Are there any scientific inaccuracies or assumptions that seem unfounded?

After your analysis, provide specific feedback on how the flip can be improved, either conceptually (changing the idea) or microscopically (changing the implementation). 
Specifically, give feedback on the frailties of the idea as a whole, and suggest potential enhancements.
"""

# Reviser Chain: Takes original hypothesis and critique, then provides a revised hypothesis.
reviser_template = """
Given the original hypothesis:
{hypothesis}

And based on the critique provided:
{critique}

Revise the hypothesis by addressing each point of the critique. Ensure the new version is clearer, more coherent, and scientifically valid. Only write the revised hypothesis, nothing else.
"""

def adversarial_update_hypothesis(hypothesis):
    critic_prompt = PromptTemplate(input_variables=["hypothesis"], template=critic_template)
    critic_chain = LLMChain(llm=llm, prompt=critic_prompt, output_key="critique")
    reviser_prompt = PromptTemplate(input_variables=["hypothesis", "critique"], template=reviser_template)
    reviser_chain = LLMChain(llm=llm, prompt=reviser_prompt, output_key="revised_hypothesis")
    overall_chain = SequentialChain(
        chains=[critic_chain, reviser_chain],
        input_variables=["hypothesis"],
        output_variables=["critique", "revised_hypothesis"],
        verbose=False)
    result = overall_chain({"hypothesis": str(hypothesis)})
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