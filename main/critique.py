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

Revise the hypothesis by addressing each point of the critique. Ensure the new version is clearer, more coherent, and scientifically valid.
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

def improve_hypothesis(hypothesis: str, n_iters: int = 3) -> str:
    for i in tqdm(range(n_iters)):
        revised_hypothesis = adversarial_update_hypothesis(hypothesis)
        hyp_str = revised_hypothesis['revised_hypothesis']
        hyp_str = hyp_str.replace('\n', ' ')
        # Remove everything before the first { and after the last }
        hyp_str = re.sub(r'^.*?\{', '{', hyp_str)
        hypothesis = hyp_str
    return hypothesis

hypothesis_to_improve = {
    'Bit': 'Statistical model estimation in sensor networks requires advanced and costly joint optimization methods for distributed learning.',
    'Flip': 'Simple linear combination or max-voting methods, when combined with second-order information, can be statistically competitive, offering low communication and computational cost and "any-time" behavior.'
}

hypothesis = improve_hypothesis(hypothesis=hypothesis_to_improve, n_iters=3)
print(hypothesis)