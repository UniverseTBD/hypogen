import json
import pandas as pd
from tqdm.notebook import tqdm

TOTAL_PAPERS = 2307752


def load_arxiv_json(author='Yuan-Sen Ting', path='arxiv.json', total_papers=TOTAL_PAPERS):
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