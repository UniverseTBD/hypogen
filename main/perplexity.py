import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

ZERO_SHOT_PROMPT = """
Consider the Bit-Flip concept:

* Bit: The prevailing belief or conventional approach in a given domain.
* Flip: The counterargument or innovative approach that challenges or overturns the 'Bit'.

Your task is to generate the 'Flip' based on the provided 'Bit' in the field of machine learning. Please articulate a 'Flip' that logically counters or innovates upon the given 'Bit'. Your 'Flip' should consist of three sentences that are logically connected, providing a holistic view of the innovative approach or counterargument.

Remember, avoid phrases like "This research aims to..." or "This paper proposes...", and instead focus on describing the innovative approach or counterargument directly. Your flip should be creative, novel, practical and elegant.

Do not provide anything else except your three sentence flip. 
"""

class PerplexityDataset:
    def __init__(self, path: str = "../data/cs.LG_test.csv"):
        self.df = pd.read_csv(path)
        self.bits = self.df['bit'].tolist()
        self.flips = self.df['flip'].tolist()

    def __len__(self):
        return len(self.bits)

    def __getitem__(self, i):
        return self.bits[i], self.flips[i]


def compute_perplexity(model, tokenizer, bit, flip):
    input_text = f"### BIT: {bit}\n ### Flip: {flip}" # ZERO_SHOT_PROMPT + f"### BIT: {bit}\n ### Flip: {flip}"
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


def main():
    # Load model and tokenizer
    model_path = "/g/data/y89/cn1951/hypogen/hypogen_llama" #"/g/data/y89/cn1951/llama-13b"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                 device_map="auto", trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataset
    dataset = PerplexityDataset()

    # Compute perplexity for each bit-flip pair in the dataset
    perplexities = []
    for i in tqdm(range(len(dataset))):
        bit, flip = dataset[i]
        perplexity = compute_perplexity(model, tokenizer, bit, flip)
        perplexities.append(perplexity)
        #print(f"Index: {i}, Bit: {bit}, Perplexity: {perplexity}")

    # Print average perplexity
    avg_perplexity = sum(perplexities) / len(perplexities)
    print(f"Average Perplexity: {avg_perplexity}")


if __name__ == "__main__":
    main()