from entropy_reducer import EnhancedQuestionOrderer
from perfect_match_orderer import PerfectMatchQuestionOrderer
import pandas as pd
import numpy as np
from typing import List


class VAASimulator:
    
    def __init__(self, responses: np.ndarray, labels: List[str], parties: List[str]):
        """
        Initialize the simulator with PerfectMatchQuestionOrderer.

        Args:
            responses: Array of historical responses (n_candidates, n_questions).
            labels: Labels for the questions.
            parties: List of parties corresponding to each candidate.
        """
        self.responses = responses
        self.labels = labels
        self.parties = parties
        self.remaining_candidates = list(range(len(parties)))  # Start with all candidates
        self.answered_questions = []
        self.algorithm_name = "PerfectMatchQuestionOrderer"
        self.orderer = PerfectMatchQuestionOrderer(responses, question_labels=labels, parties=parties, verbose=True)

    def filter_candidates_by_party(self):
        """
        Allow the user to choose whether to include or exclude candidates based on party preferences.
        """
        print("List of Parties:")
        for i, party in enumerate(set(self.parties)):
            print(f"{i + 1}. {party}")

        print("\nDo you want to:")
        print("1. Include only certain parties")
        print("2. Exclude certain parties")
        print("3. Skip party filtering")
        
        while True:
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice in ["1", "2", "3"]:
                break
            print("Invalid input! Please enter '1', '2', or '3'.")

        if choice == "1":
            # Include specific parties
            include_input = input("Enter the numbers of the parties you WILL vote for (comma-separated): ").strip()
            include_indices = [int(x) - 1 for x in include_input.split(",") if x.isdigit()]
            include_parties = [list(set(self.parties))[i] for i in include_indices]

            self.remaining_candidates = [
                i for i in self.remaining_candidates if self.parties[i] in include_parties
            ]

            # Print included parties
            print(f"\nYou chose to include the following parties: {', '.join(include_parties)}")

        elif choice == "2":
            # Exclude specific parties
            exclude_input = input("Enter the numbers of the parties you WILL NOT vote for (comma-separated): ").strip()
            exclude_indices = [int(x) - 1 for x in exclude_input.split(",") if x.isdigit()]
            exclude_parties = [list(set(self.parties))[i] for i in exclude_indices]

            self.remaining_candidates = [
                i for i in self.remaining_candidates if self.parties[i] not in exclude_parties
            ]
            
            # Print excluded parties
            print(f"\nYou chose to exclude the following parties: {', '.join(exclude_parties)}")

        else:
            # Skip party filtering
            print("No party filtering applied.")

        # Update the PerfectMatchQuestionOrderer with the filtered candidates
        self.orderer.remaining_candidates = self.remaining_candidates

        print(f"\nFiltered candidates based on party preferences. {len(self.remaining_candidates)} candidates remain.")

    def run_simulation(self):
        """
        Run the command-line simulation using PerfectMatchQuestionOrderer.
        """
        print(f"\n--- VAA Simulator using {self.algorithm_name} ---")

        # Step 1: Filter candidates by party preferences
        self.filter_candidates_by_party()

        # Step 2: Run the Perfect Match algorithm
        self.orderer.run_algorithm(max_matches=5)

        print("\n--- Simulation Complete ---")
        print("Your answers and results have been recorded.")

# Main simulation script
if __name__ == "__main__":
    # Load and preprocess real data
    data = pd.read_csv('data.csv')
    data = data.iloc[:, [1]].join(data.iloc[:, 3:32])  # Use only relevant columns

    # Handle missing data
    data.replace('-', np.nan, inplace=True)
    nan_threshold = (len(data.columns) - 1) * 0.9  # Drop rows with too many NaNs
    data = data.dropna(thresh=nan_threshold, subset=data.columns[1:])
    data.fillna(3, inplace=True)  # Fill missing values with neutral (3)

    # Extract parties
    parties = data.iloc[:, 0].tolist()

    # Convert responses to numeric and extract labels
    answer_vectors = data.iloc[:, 1:].values
    responses = answer_vectors.astype(int)
    labels = list(data.columns[1:])

    # Choose algorithm and initialize simulator
    simulator = VAASimulator(responses, labels, parties=parties)
    simulator.run_simulation()
