# Import necessary modules and libraries
import yaml
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

# Load the configurations
config = yaml.safe_load(open("../config.yml"))
API_KEY = config['api_key']
DEPLOYMENT_NAME = config['deployment_name']
BASE_URL = config['base_url']
API_VERSION = config['api_version']

# Initialize the model
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

# Define the function to expand the flip
def expand_flip(flip: str) -> str:
    expansion_prompt = PromptTemplate(input_variables=["flip"], template=expansion_template)
    expansion_chain = LLMChain(llm=llm, prompt=expansion_prompt, output_key="expanded_flip")
    result = expansion_chain({"flip": flip})
    return result['expanded_flip']
