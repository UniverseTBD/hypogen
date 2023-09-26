# Import required modules for hypogen.py
import json
import csv
from tqdm import tqdm  # Import tqdm for progress bars
from tournament import run_knockout_tournament  # Importing from the provided tournament.py script
from critique import improve_hypothesis  # Importing from the provided critique.py script
import logging
import os

# Create logging directory if it doesn't exist
log_dir = "../logs/hypogen"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=f'{log_dir}/hypogen.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hypogen_main(json_file_path: str, csv_output_path: str, n_iters: int = 3):
    """
    Main function to run the hypogen process.

    Parameters:
    - json_file_path: str : The file path for the JSON containing the proposal hypotheses.
    - csv_output_path: str : The output file path for saving the final flip in CSV format.
    - n_iters: int : Number of iterations for the critique and refinement process.
    """
    print("Loading proposal data...")
    # Load the JSON file containing proposal hypotheses
    with open(json_file_path, 'r') as f:
        proposal_data = json.load(f)
    
    print("Preparing CSV output...")
    # Create or open the CSV file to save the final flips
    with open(csv_output_path, 'a', newline='') as csvfile:
        fieldnames = ['Bit', 'Flip', 'Final']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file is empty
        if os.stat(csv_output_path).st_size == 0:
            writer.writeheader()
        
        print("Processing bits...")
        # Iterate through each bit in the JSON
        for bit in tqdm(proposal_data.keys()):  # Using tqdm for progress bar
            proposals = proposal_data[bit]
            logging.info(f"Processing bit: {bit}")
            
            print(f"Running knockout tournament for bit: {bit}")
            # Run the knockout tournament to get the winning flip
            winner = run_knockout_tournament(bit, proposals)
            logging.info(f"The winner for bit {bit} is: {winner}")
            
            print(f"Improving the winning hypothesis for bit: {bit}")
            # Improve the winning hypothesis through critique and refinement
            hypothesis_to_improve = {'Bit': bit, 'Flip': winner}
            improved_hypothesis = improve_hypothesis(hypothesis=hypothesis_to_improve, n_iters=n_iters)
            print(f"Final improved hypothesis for bit: {bit}: {improved_hypothesis}")
            logging.info(f"Final improved hypothesis: {improved_hypothesis}")
            
            print(f"Saving the final hypothesis for bit: {bit}")
            # Save the final flip to CSV
            writer.writerow(improved_hypothesis)

if __name__ == "__main__":
    print("Starting HypoGen main process...")
    hypogen_main(
        json_file_path="../data/generated/proposal_hypogen.json",  # Replace with your actual JSON path
        csv_output_path="../data/generated/hypogen.csv"  # Replace with your desired CSV output path
    )
    print("HypoGen main process completed.")