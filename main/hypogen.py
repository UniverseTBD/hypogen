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

def hypogen_main(json_file_path, csv_output_path, n_iters=3):
    print("Loading proposal data...")
    with open(json_file_path, 'r') as f:
        proposal_data = json.load(f)

    last_bit = None
    if os.path.exists(csv_output_path):
        df_existing = pd.read_csv(csv_output_path)
        if not df_existing.empty:
            last_bit = df_existing['Bit'].iloc[-1]
            print(f"Last processed bit was: {last_bit}. Starting from the next bit.")

    if last_bit:
        all_bits = list(proposal_data.keys())
        last_bit_index = all_bits.index(last_bit)
        remaining_bits = all_bits[last_bit_index + 1:]
        proposal_data = {bit: proposal_data[bit] for bit in remaining_bits}

    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(worker, proposal_data.items()), total=len(proposal_data)):
            if result:
                results.append(result)

    df_results = pd.DataFrame(results)
    
    if os.path.exists(csv_output_path):
        df_existing = pd.read_csv(csv_output_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)

    df_results.to_csv(csv_output_path, index=False)
    print("HypoGen main process completed.")

if __name__ == "__main__":
    print("Starting HypoGen main process...")
    hypogen_main(
        json_file_path="../data/generated/proposal_hypogen.json",
        csv_output_path="../data/generated/hypogen.csv"
    )
