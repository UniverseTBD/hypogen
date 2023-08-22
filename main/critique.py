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

Analyze each part of the hypothesis:
1. Clarity: Are the statements clear and easy to understand?
2. Coherence: Do the different parts of the hypothesis logically flow together? Does it seem like the proposed solution would work?
3. Scientific Validity: Are there any scientific inaccuracies or assumptions that seem unfounded?

After your analysis, provide specific feedback on how each field (Problem, Solution, Methodology, Evaluation, Results) can be improved.
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
    "Problem": "Issues in determining basic characteristics of black holes and their surrounding disks in X-ray binary, using models of the source's disk X-ray continuum. A key issue is the determination of the \"color correction factor\".",
    "Solution": "Using observational data to estimate the color correction factor by modeling the disk spectrum with saturated Compton scattering.",
    "Methodology": "The work is based on two observations made by XMM-Newton on GX 339-4. These observations offer high-quality data at low energies. The spectra were then fitted to these models.",
    "Evaluation": "The quality of fit of the spectra to the models was examined. Other models were also tested for fit.",
    "Results": "The spectra fits well with the model and provides reasonable values for the color correction factor. However, the high-soft-state continuum cannot be adequately fitted by the latest disk models."
}

hypothesis = improve_hypothesis(hypothesis=hypothesis_to_improve, n_iters=3)
print(hypothesis)