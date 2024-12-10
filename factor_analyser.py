import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

class FactorAnalyzer:
    def __init__(self, verbose=True, variance_threshold=0.1):
        self.verbose = verbose
        self.variance_threshold = variance_threshold

    def _parallel_analysis(self, n_iterations=100):
        """Determine significant factors using parallel analysis"""
        n_samples, n_features = self.standardized_responses.shape
        
        fa = FactorAnalysis(n_components=min(n_features, n_samples))
        fa.fit(self.standardized_responses)
        actual_evs = np.var(fa.transform(self.standardized_responses), axis=0)
        
        random_evs = np.zeros((n_iterations, min(n_features, n_samples)))
        for i in range(n_iterations):
            random_data = np.random.normal(size=(n_samples, n_features))
            fa.fit(random_data)
            random_evs[i] = np.var(fa.transform(random_data), axis=0)
        
        random_ev_95 = np.percentile(random_evs, 95, axis=0)
        n_factors = sum(actual_evs > random_ev_95)
        
        if self.verbose:
            print(f"Parallel analysis suggests {n_factors} significant factors")
            
        return n_factors, actual_evs

    def fit(self, responses, parties=None):
        if self.verbose:
            print(f"\nInitializing factor analysis with {responses.shape[1]} variables and {responses.shape[0]} responses")
        
        self.parties = parties
        self.scaler = StandardScaler()
        self.standardized_responses = self.scaler.fit_transform(responses)
        
        self.n_factors, self.eigenvalues = self._parallel_analysis()
        
        if self.n_factors == 0:
            raise ValueError("No significant factors found. Try with raw data or different preprocessing.")
            
        self.fa = FactorAnalysis(n_components=self.n_factors, random_state=42)
        self.fa.fit(self.standardized_responses)
        
        self.factor_scores = self.fa.transform(self.standardized_responses)
        self.loadings = self.fa.components_.T
        
        self.explained_variance = np.var(self.factor_scores, axis=0)
        self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)
        
        self.significant_factors = np.where(self.explained_variance_ratio > self.variance_threshold)[0]
        
        if self.verbose:
            print(f"Factor analysis complete. Found {len(self.significant_factors)} factors explaining >{self.variance_threshold*100}% variance each")
            
        return self
    
    def print_significant_questions(self, labels, top_n=8):  # Increased from 4 to 8
        """Print the most significant questions for each factor"""
        for factor_idx in self.significant_factors:
            factor_loadings = self.loadings[:, factor_idx]
            # Get indices of top n questions by absolute loading value
            top_indices = np.argsort(np.abs(factor_loadings))[-top_n:]
            
            print(f"\nFactor {factor_idx + 1} (explains {self.explained_variance_ratio[factor_idx]*100:.1f}% variance)")
            print("Most significant questions:")
            print(f"{'Question':<50} | {'Loading':>8} | {'Abs Loading':>11}")
            print("-" * 73)
            for idx in reversed(top_indices):
                print(f"{labels[idx][:50]:<50} | {factor_loadings[idx]:8.3f} | {abs(factor_loadings[idx]):11.3f}")
            print()  # Add blank line between factors
    
    def plot_individual_factors(self):
        """Plot each significant factor pair individually"""
        factor_pairs = list(combinations(self.significant_factors, 2))
        
        for i, j in factor_pairs:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(self.factor_scores[:, i], 
                                self.factor_scores[:, j], 
                                c=[plt.cm.tab20(hash(p) % 20) for p in self.parties] if self.parties is not None else None,
                                alpha=0.6)
            
            variance_i = self.explained_variance_ratio[i] * 100
            variance_j = self.explained_variance_ratio[j] * 100
            plt.xlabel(f'Factor {i+1} ({variance_i:.1f}% var)')
            plt.ylabel(f'Factor {j+1} ({variance_j:.1f}% var)')
            plt.grid(True)
            
            if self.parties is not None:
                # Add legend with unique parties
                unique_parties = list(set(self.parties))
                handles = [plt.scatter([], [], c=plt.cm.tab20(hash(p) % 20), label=p) for p in unique_parties]
                plt.legend(handles=handles, title='Parties', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()

    def get_factor_importance(self):
            """Calculate and return the explained variance ratio for each factor"""
            # Create array of False values initially
            is_significant = np.zeros(self.n_factors, dtype=bool)
            # Set True for significant factors
            is_significant[self.significant_factors] = True
            
            importance_df = pd.DataFrame({
                'Factor': [f'Factor_{i+1}' for i in range(self.n_factors)],
                'Explained_Variance_Ratio': self.explained_variance_ratio,
                'Cumulative_Variance_Ratio': np.cumsum(self.explained_variance_ratio),
                'Significant': is_significant
            })
            return importance_df

# Data preprocessing
data = pd.read_csv('data.csv')
data = data.iloc[:, [1]].join(data.iloc[:, 3:32])

data.replace('-', np.nan, inplace=True)
nan_threshold = (len(data.columns) - 1) * 0.8
data = data.dropna(thresh=nan_threshold, subset=data.columns[1:])
data.fillna(3, inplace=True)

parties = data.iloc[:,0].tolist()  # Fixed toList() to tolist()

answer_vectors = data.iloc[:, 1:].values
responses = answer_vectors.astype(int)
labels = list(data.columns[1:])

# Analysis
analyzer = FactorAnalyzer(verbose=True, variance_threshold=0.1)
analyzer.fit(responses, parties)

# Print factor scores (fixed line)
print("\nFactor scores for each response (significant factors only):")
print(analyzer.factor_scores[:5, analyzer.significant_factors])  # Changed from factor_scores to analyzer.factor_scores

# Print significant questions for each factor
analyzer.print_significant_questions(labels)

# Plot individual factor pairs
analyzer.plot_individual_factors()

# Print factor importance
print("\nFactor importance:")
print(analyzer.get_factor_importance())

# Print factor importance
print("\nFactor importance:")
print(analyzer.get_factor_importance())