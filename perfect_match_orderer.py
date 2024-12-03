import numpy as np
import pandas as pd
from typing import List, Optional


class PerfectMatchQuestionOrderer:
    def __init__(self, responses: np.ndarray, question_labels: Optional[List[str]] = None,
                 parties: Optional[List[str]] = None, verbose: bool = True):
        """
        Initialize with historical responses and question labels.

        Args:
            responses: Array of shape (n_candidates, n_questions) with candidate answers.
            question_labels: List of question labels for display purposes.
            parties: List of parties corresponding to each candidate.
            verbose: Whether to print detailed information.
        """
        self.responses = responses
        self.n_candidates, self.n_questions = responses.shape
        self.verbose = verbose
        self.question_labels = question_labels or [f"Q{i}" for i in range(self.n_questions)]
        self.parties = parties
        self.remaining_candidates = list(range(self.n_candidates))
        self.answered_questions = []

        if self.verbose:
            print(f"PerfectMatch initialized with {self.n_candidates} candidates and {self.n_questions} questions.")

    def calculate_question_entropy(self, question_idx: int, candidates: List[int]) -> float:
        """
        Calculate the entropy of the responses for a specific question among remaining candidates.

        Args:
            question_idx: Index of the question.
            candidates: List of candidate indices still under consideration.

        Returns:
            Entropy of the responses for the question.
        """
        responses = self.responses[candidates, question_idx]
        counts = np.bincount(responses, minlength=6)  # Ensure all options (1-5) are counted
        probabilities = counts / len(candidates)
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def select_next_question(self) -> Optional[int]:
        """
        Select the next question with the highest entropy among the remaining candidates.

        Returns:
            Index of the selected question, or None if no questions remain.
        """
        if not self.remaining_candidates:
            raise ValueError("No remaining candidates to evaluate.")

        # Filter questions that have not been answered yet
        available_questions = [
            q for q in range(self.n_questions) if q not in self.answered_questions
        ]

        if not available_questions:
            if self.verbose:
                print("No more unanswered questions available.")
            return None

        # Calculate entropy for available questions
        entropies = [
            self.calculate_question_entropy(q, self.remaining_candidates)
            for q in available_questions
        ]

        # Select the question with the maximum entropy
        best_question = available_questions[np.argmax(entropies)]
        if self.verbose:
            print(f"\nSelected question {best_question} with max entropy.")
        return best_question

    def filter_candidates(self, question_idx: int, answer: int, filtering_mode: str = "exact"):
        """
        Filter the candidates based on the user's answer to a question.

        Args:
            question_idx: Index of the question answered.
            answer: User's answer to the question.
            filtering_mode: The filtering algorithm to use ("exact" for standard, "range" for the new logic).
        """
        if filtering_mode == "exact":
            # Standard filtering: keep candidates whose answer matches the user's answer
            self.remaining_candidates = [
                c for c in self.remaining_candidates if self.responses[c, question_idx] == answer
            ]
        elif filtering_mode == "range":

            # New filtering logic based on max difference of 1
            self.remaining_candidates = [
                c for c in self.remaining_candidates if abs(self.responses[c, question_idx] - answer) <= 1
            ]
        self.answered_questions.append(question_idx)

        if self.verbose:
            print(f"\nFiltered candidates. {len(self.remaining_candidates)} candidates remain.")


    def display_response_counts(self, question_idx: int):
        """
        Display the counts of each response (1-5) for the selected question.

        Args:
            question_idx: Index of the question.
        """
        responses = self.responses[self.remaining_candidates, question_idx]
        counts = np.bincount(responses, minlength=6)[1:6]  # Count responses for 1-5
        print(f"\nResponse counts for question '{self.question_labels[question_idx]}':")
        for i, count in enumerate(counts, start=1):
            print(f"  {i}: {count}")

    def select_filtering_mode(self):
        """
        Prompt user to select a filtering mode and validate the choice.

        Returns:
            A string representing the selected filtering mode ("exact" or "range").
        """
        print("\nChoose a filtering mode:")
        print("1. Exact match (candidates must answer exactly the same as you)")
        print("2. Range-based (candidates must have a maximum difference of 1 from your answer)")
        
        while True:
            choice = input("Enter the number corresponding to your choice (1/2): ").strip()
            if choice == "1":
                return "exact"
            elif choice == "2":
                return "range"
            print("Invalid input! Please enter '1' or '2'.\n")


    def run_algorithm(self, max_matches: int = 5) -> List[int]:
        """
        Run the "Perfect Match" algorithm until the number of remaining candidates is below the threshold.

        Args:
            max_matches: Number of matching candidates to stop at.

        Returns:
            List of remaining candidate indices.
        """
        filtering_mode = self.select_filtering_mode()

        while len(self.remaining_candidates) > max_matches:
            next_question = self.select_next_question()
            self.display_response_counts(next_question)
            print(f"\n{self.question_labels[next_question]} (1: Totally Disagree - 5: Totally Agree)")
            while True:
                try:
                    user_response = int(input("Your answer (1-5): ").strip())
                    if user_response not in [1, 2, 3, 4, 5]:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid input! Please enter a number between 1 and 5.")
            self.filter_candidates(next_question, user_response, filtering_mode=filtering_mode)

        if self.verbose:

            def print_candidate(candidate_index):
                party = self.parties[candidate_index] if self.parties else "Unknown"
                print(f"Candidate {candidate_index} (Party: {party})")

            print("\nAlgorithm complete. Remaining candidates:")
            for candidate_index in self.remaining_candidates:
                print_candidate(candidate_index)

            print("\nAnswers of remaining candidates:")
            for candidate_index in self.remaining_candidates:
                print()
                print_candidate(candidate_index)
                
                # Print candidate's answers for all questions
                answers = self.responses[candidate_index]
                for question, answer in zip(self.question_labels, answers):
                    print(f"{question}: {answer}")

        return self.remaining_candidates

