from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import mauve
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import openai
import yaml
from nltk.util import ngrams
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings.openai import OpenAIEmbeddings

class TableEvaluation:

    def __init__(self, method: str):
        self.method = method
        self.path = "../data/generated"
        self.synthetic_dataset_path = os.path.join(self.path, f"{self.method}.csv")
        self._load_datasets()
        self.deployment_name = setup_openai()
        self.embeddings = self.setup_embeddings()
    
    def _load_datasets(self):
        self.synthetic_dataset = pd.read_csv(self.synthetic_dataset_path)['flip'].tolist()
        self.real_dataset = pd.read_csv(self.path+'/real.csv')['flip'].tolist()

    def set_embeddings(self, local_disk=True):
        real_embeddings_path = "../data/embeddings/real.npy"
        synthetic_embeddings_path = f"../data/embeddings/{self.method}-synthetic.npy"
        
        if local_disk and os.path.exists(synthetic_embeddings_path):
            self.synthetic_embeddings = np.load(synthetic_embeddings_path)
        else:
            self.synthetic_embeddings = self.embed_dataset(self.synthetic_dataset)
            np.save(synthetic_embeddings_path, self.synthetic_embeddings)
        
        if not os.path.exists(real_embeddings_path) or not local_disk:
            self.real_embeddings = self.embed_dataset(self.real_dataset)
            np.save(real_embeddings_path, self.real_embeddings)
        else:
            self.real_embeddings = np.load(real_embeddings_path)

    def setup_embeddings(self):
        config = yaml.safe_load(open("../config.yml", "r"))
        API_KEY = config['embedding_api_key']
        DEPLOYMENT_NAME = "embedding"
        BASE_URL = config['embedding_base_url']
        os.environ['OPENAI_API_KEY'] = API_KEY

        embeddings = OpenAIEmbeddings(
            openai_api_base=BASE_URL,
            deployment=DEPLOYMENT_NAME,
            model="text-embedding-ada-002",
            openai_api_key=API_KEY,
            openai_api_type="azure",
            chunk_size=64,
        )
        return embeddings
    
    def worker(self, text):
        if text is None:
            return None
        embedded_text = self.embeddings.embed_query(text)
        return (text, embedded_text)
    
    def embed_dataset(self, dataset) -> np.array:
        text_embeddings = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(tqdm(executor.map(self.worker, dataset), total=len(dataset), desc="Computing Embeddings"))
            text_embeddings.extend([(text, embedding) for text, embedding in results if text and embedding])
        
        _, embeddings = zip(*text_embeddings)
        return np.array(embeddings)

    def mauve(self):
        mauve_results = mauve.compute_mauve(p_features=self.real_embeddings, q_features=self.synthetic_embeddings,
                                            verbose=False)
        return mauve_results.mauve
    
    def authenticity_auroc(self):
        # Replace with the actual method from your provided code
        # Create labels
        real_labels = np.ones(len(self.real_embeddings))
        synthetic_labels = np.zeros(len(self.synthetic_embeddings))

        # Combine the data
        data = np.vstack((self.real_embeddings, self.synthetic_embeddings))
        labels = np.concatenate((real_labels, synthetic_labels))

        # Create the k-fold cross-validation object
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        roc_auc_scores = []
        
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            model = xgb.XGBClassifier()
            model.fit(X_train, y_train)

            roc_auc_scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            
        return np.mean(roc_auc_scores)

    def evaluate(self):
        self.set_embeddings()
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
        
        results = {
            'n_grams': normalised_ngrams(self.synthetic_dataset, self.real_dataset, tokenizer, 3),
            'diversity': diversity(self.synthetic_dataset, self.real_dataset, tokenizer),
            'mauve': self.mauve(),
            'auroc': self.authenticity_auroc()
        }
        
        return results

def setup_openai():
    config = yaml.safe_load(open("../config.yml", "r"))
    openai.api_key = config['embedding_api_key']
    openai.api_base = config['embedding_base_url']
    return "embeddings"

def normalised_ngrams(synthetic_dataset, real_dataset, tokenizer, n) -> dict:
    results = {}
    synthetic_text = " ".join(synthetic_dataset)
    real_text = " ".join(real_dataset)[:len(synthetic_text)]
    for text, label in zip([synthetic_text, real_text], ['synthetic', 'real']):
        tokens = tokenizer.tokenize(text)
        generated_ngrams = list(ngrams(tokens, n))
        unique_ngrams = len(set(generated_ngrams))
        results[label] = 100 * (1 - (unique_ngrams / len(generated_ngrams))) if generated_ngrams else 0
    return results

def diversity(synthetic_dataset, real_dataset, tokenizer) -> dict:
    results = {}
    for dataset, label in zip([synthetic_dataset, real_dataset], ['synthetic', 'real']):
        norm_values_product = 1.0
        text = " ".join(dataset)
        for n in range(2, 5):
            tokens = tokenizer.tokenize(text)
            generated_ngrams = list(ngrams(tokens, n))
            unique_ngrams = len(set(generated_ngrams))
            norm_n = 1 - unique_ngrams / len(generated_ngrams) if generated_ngrams else 0
            norm_values_product *= norm_n
        results[label] = norm_values_product
    return results

if __name__ == "__main__":
    method = "gpt_4_zero_shot"
    evaluator = TableEvaluation(method)
    results = evaluator.evaluate()
    print(f"Results for {method}:")
    print(results)