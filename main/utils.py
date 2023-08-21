import json
import pandas as pd
from tqdm import tqdm

TOTAL_PAPERS = 2307752


def load_arxiv_json_by_author(author='Yuan-Sen Ting', path='arxiv.json', total_papers=TOTAL_PAPERS):
    data = []
    with open(path, 'r') as f:
        for line in tqdm(f, total=total_papers):
            try:
                json_line = json.loads(line)
                if author in json_line.get('authors', ''):
                    data.append(json_line)
            except json.JSONDecodeError: continue

    df = pd.DataFrame(data)
    df.to_csv('arxiv.csv', index=False)
    return df

def load_arxiv_json_by_category(category='astro-ph', path='arxiv.json', total_papers=TOTAL_PAPERS):
    data = []
    with open(path, 'r') as f:
        for line in tqdm(f, total=total_papers):
            try:
                json_line = json.loads(line)
                if category in json_line.get('categories', ''):
                    data.append(json_line)
            except json.JSONDecodeError: continue

    df = pd.DataFrame(data)
    df.to_csv('arxiv.csv', index=False)
    return df

def preprocess_for_finetuning(path: str = 'extracted.json'):
    # Read in the JSON file
    with open(path, 'r') as f: examples = json.load(f)
    # Create a list of texts
    output_texts = []
    for i in range(len(examples)):
        text = {'text': f"### Instruction: Generate a hypothesis about the following problem: {examples[i]['Problem']}\n \
                 ### Hypothesis\n: Problem: {examples[i]['Problem']}\n \
                 Solution: {examples[i]['Solution']}\n \
                 Methodology: {examples[i]['Methodology']}\n \
                 Evaluation: {examples[i]['Evaluation']}\n \
                 Results: {examples[i]['Results']}"}
        output_texts.append(text)
    # Wipe the current JSON file
    with open('extracted_preprocessed.json', 'w') as f: json.dump([], f)
    # Save the texts to a JSON file
    with open('extracted_preprocessed.json', 'w') as f: json.dump(output_texts, f)

if __name__ == '__main__':
    load_arxiv_json_by_category()
    # preprocess_for_finetuning()