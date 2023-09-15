import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

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

def load_arxiv_json_by_category(category='astro-ph', path='../data/raw/arxiv.json', total_papers=TOTAL_PAPERS):
    data = []
    with open(path, 'r') as f:
        for line in tqdm(f, total=total_papers):
            try:
                json_line = json.loads(line)
                if category in json_line.get('categories', ''):
                    data.append(json_line)
            except json.JSONDecodeError: continue

    df = pd.DataFrame(data)
    columns = ['title', 'categories', 'abstract', 'authors', 'doi', 'id']
    df = df[columns]
    df.to_csv(f'../data/processed/arxiv-{category}.csv', index=False)
    return df

def preprocess_for_finetuning(path: str = '../data/processed/yuan_train.csv'):
    # Read in the CSV
    df = pd.read_csv(path)
    bits = df['bit'].values
    flips = df['flip'].values
    # Create a list of texts
    output_texts = []
    for i in range(len(df)):
        text = {'text': f"### Instruction: Generate a proposed hypothesis in the bit-flip schema:\
                 ### BIT\n: Problem: {bits[i]}\n \
                 FLIP: {flips[i]}"}
        output_texts.append(text)
    # Wipe the current JSON file
    with open('train.json', 'w') as f: json.dump([], f)
    # Save the texts to a JSON file
    with open('train.json', 'w') as f: json.dump(output_texts, f)

def upload_dataset_to_hf(category: str):
    df = pd.read_csv(f"../data/tuning/{category}.csv")
    print(f"There are {len(df)} hypotheses in the dataset.")
    abstracts = pd.read_csv(f"../data/processed/arxiv-{category}.csv", low_memory=False)
    print(f"There are {len(abstracts)} abstracts in the dataset.")
    # Merge the df and abstracts dataframes on title
    merged_df = pd.merge(df, abstracts, on='title')
    merged_df.drop(columns=['index'], inplace=True)
    print(f"There are {len(merged_df)} hypotheses in the merged dataset.")
    merged_df.to_csv(f"../data/processed/hf-{category}.csv", index=False)
    # Load as HF dataset and push to hub
    dataset = load_dataset('csv', data_files=f"../data/processed/hf-{category}.csv", split='train')
    dataset.push_to_hub(f"universeTBD/arxiv-bit-flip-{category}")
    print(f"Dataset uploaded to HF hub: universeTBD/arxiv-bit-flip-{category}")

if __name__ == '__main__':
    load_arxiv_json_by_category(category='cs.LG')