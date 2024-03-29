import json
import pandas as pd
from tqdm import tqdm
from tournament import run_knockout_tournament
from critique import improve_hypothesis
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import os

def process_bit(bit, proposals, n_iters):
    print(f"Processing bit: {bit}")
    start = time.time()
    winner = run_knockout_tournament(bit, proposals)
    end = time.time()
    print(f"Time taken for tournament: {end - start:.2f} seconds")
    hypothesis_to_improve = {'Bit': bit, 'Flip': winner}
    start = time.time()
    improved_hypothesis = improve_hypothesis(hypothesis=hypothesis_to_improve, n_iters=n_iters)
    end = time.time()
    print(f"Time taken for improvement: {end - start:.2f} seconds")
    return improved_hypothesis

def worker(bit_proposals_tuple):
    try:
        bit, proposals = bit_proposals_tuple
        return process_bit(bit, proposals, n_iters=2)
    except Exception as e:
        logging.error(f"Error processing bit: {bit}. Error: {e}")
        return None

def hypogen_main(json_file_path, csv_output_path, n_bits, start_index=0):
    print("Loading proposal data...")
    with open(json_file_path, 'r') as f:
        proposal_data = json.load(f)

    all_bits = list(proposal_data.keys())[start_index:start_index + n_bits]
    subset_proposal_data = {bit: proposal_data[bit] for bit in all_bits}

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(worker, subset_proposal_data.items()), total=len(subset_proposal_data)):
            if result:
                results.append(result)

    df_results = pd.DataFrame(results)
    
    if os.path.exists(csv_output_path):
        df_existing = pd.read_csv(csv_output_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)

    df_results.to_csv(csv_output_path, index=False)
    print("Round completed.")

if __name__ == "__main__":
    num_rounds = 490
    num_bits = 2  # Number of bits to process in each run

    # Determine the starting index
    csv_output_path = "../data/generated/hypogen.csv"
    start_index = 0
    if os.path.exists(csv_output_path):
        df_existing = pd.read_csv(csv_output_path)
        if not df_existing.empty:
            last_bit = df_existing['Bit'].iloc[-1]
            with open("../data/generated/proposal_hypogen.json", 'r') as f:
                all_bits = list(json.load(f).keys())
            start_index = all_bits.index(last_bit) + 1

    for _ in tqdm(range(num_rounds), desc="Overall Progress"):
        hypogen_main(
            json_file_path="../data/generated/proposal_hypogen.json",
            csv_output_path=csv_output_path,
            n_bits=num_bits,
            start_index=start_index
        )
        start_index += num_bits
        # Clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')