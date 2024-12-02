from entropy_reducer import EnhancedQuestionOrderer
from perfect_match_orderer import PerfectMatchQuestionOrderer
import pandas as pd
import numpy as np
from typing import List


class VAASimulator:
    def __init__(self, responses: np.ndarray, labels: List[str], parties: List[str], algorithm: str):
        """
        Initialize the simulator with the chosen algorithm.

        Args:
            responses: Array of historical responses (n_candidates, n_questions).
            labels: Labels for the questions.
            algorithm: The algorithm to use ("1" for EnhancedQuestionOrderer, "2" for PerfectMatchQuestionOrderer).
        """
        self.responses = responses
        self.labels = labels
        self.parties = parties
        self.answered_questions = []

        if algorithm == "1":
            self.algorithm_name = "EnhancedQuestionOrderer"
            self.orderer = EnhancedQuestionOrderer(responses, question_labels=labels, verbose=True)
        elif algorithm == "2":
            self.algorithm_name = "PerfectMatchQuestionOrderer"
            self.orderer = PerfectMatchQuestionOrderer(responses, question_labels=labels, parties=parties, verbose=True)
        else:
            raise ValueError(f"Unknown algorithm selection: {algorithm}")

    def run_simulation(self):
        """
        Run the command-line simulation using the chosen algorithm.
        """
        print(f"\n--- VAA Simulator using {self.algorithm_name} ---")
        if isinstance(self.orderer, EnhancedQuestionOrderer):
            ordered_questions = self.orderer.order_questions()
            for q_idx in ordered_questions:
                question = self.labels[q_idx]
                print(f"\n{question} (1: Totally Disagree - 5: Totally Agree)")
                while True:
                    try:
                        response = int(input("Your answer (1-5): ").strip())
                        if response not in [1, 2, 3, 4, 5]:
                            raise ValueError
                        break
                    except ValueError:
                        print("Invalid input! Please enter a number between 1 and 5.")
                self.answered_questions.append((q_idx, response))
        elif isinstance(self.orderer, PerfectMatchQuestionOrderer):
            self.orderer.run_algorithm(max_matches=5)

        print("\n--- Simulation Complete ---")
        print("Your answers and results have been recorded.")


def select_algorithm():
    """
    Prompt user to select an algorithm and validate the choice.

    Returns:
        A string representing the selected algorithm ("1" or "2").
    """
    print("Choose an algorithm:")
    print("1. EnhancedQuestionOrderer (Entropy-based)")
    print("2. PerfectMatchQuestionOrderer (Perfect Match)")

    while True:
        choice = input("Enter the number corresponding to your choice (1/2): ").strip()
        if choice in ["1", "2"]:
            return choice
        print("Invalid input! Please enter '1' or '2'.\n")


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

    parties = data.iloc[:, 0].tolist()

    # Convert responses to numeric and extract labels
    answer_vectors = data.iloc[:, 1:].values
    responses = answer_vectors.astype(int)
    labels = list(data.columns[1:])

    # Choose algorithm and initialize simulator
    algorithm_choice = select_algorithm()
    simulator = VAASimulator(responses, labels, parties=parties, algorithm=algorithm_choice)
    simulator.run_simulation()
