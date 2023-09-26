# Import required modules for hypogen.py
import json
import csv
from tqdm import tqdm  # Import tqdm for progress bars
from tournament import run_knockout_tournament  # Importing from the provided tournament.py script
from critique import improve_hypothesis  # Importing from the provided critique.py script
import logging
import os
import time  # Import time module for timing

# Create logging directory if it doesn't exist
log_dir = "../logs/hypogen"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=f'{log_dir}/hypogen.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_bit(bit, proposals, n_iters, writer):
    logging.info(f"Processing bit: {bit}")

    # Time tracking for tournament
    start_time_tournament = time.time()
    winner = run_knockout_tournament(bit, proposals)
    end_time_tournament = time.time()
    print(f"Time taken for knockout tournament for bit {bit}: {end_time_tournament - start_time_tournament:.2f} seconds")
    
    logging.info(f"The winner for bit {bit} is: {winner}")

    # Time tracking for critique
    start_time_critique = time.time()
    hypothesis_to_improve = {'Bit': bit, 'Flip': winner}
    improved_hypothesis = improve_hypothesis(hypothesis=hypothesis_to_improve, n_iters=n_iters)
    end_time_critique = time.time()
    print(f"Time taken for critique and refinement for bit {bit}: {end_time_critique - start_time_critique:.2f} seconds")

    logging.info(f"Final improved hypothesis: {improved_hypothesis}")

    # Save to CSV
    writer.writerow(improved_hypothesis)

def hypogen_main(json_file_path: str, csv_output_path: str, n_iters: int = 3):
    print("Loading proposal data...")
    with open(json_file_path, 'r') as f:
        proposal_data = json.load(f)
    
    print("Preparing CSV output...")
    with open(csv_output_path, 'a', newline='') as csvfile:
        fieldnames = ['Bit', 'Flip', 'Final']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if os.stat(csv_output_path).st_size == 0:
            writer.writeheader()

        print("Processing bits...")
        for bit, proposals in tqdm(proposal_data.items()):
            process_bit(bit, proposals, n_iters, writer)

if __name__ == "__main__":
    print("Starting HypoGen main process...")
    hypogen_main(
        json_file_path="../data/generated/proposal_hypogen.json",  # Replace with your actual JSON path
        csv_output_path="../data/generated/hypogen.csv"  # Replace with your desired CSV output path
    )
    print("HypoGen main process completed.")