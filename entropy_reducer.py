import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class EnhancedQuestionOrderer:
    def __init__(self, historical_responses: np.ndarray, n_latent_factors: int = None, 
                 question_labels: Optional[List[str]] = None, verbose: bool = True):
        """
        Initialize with historical response data and automatically determine optimal
        number of latent factors if not specified
        
        Args:
            historical_responses: array of shape (n_respondents, n_questions)
            n_latent_factors: number of latent factors to use (optional)
            question_labels: list of question labels for visualization (optional)
            verbose: whether to print detailed information
        """
        self.responses = historical_responses
        self.n_questions = historical_responses.shape[1]
        self.verbose = verbose
        self.question_labels = question_labels or [f"Q{i}" for i in range(self.n_questions)]
        
        if self.verbose:
            print(f"\nInitializing with {self.n_questions} questions and {len(historical_responses)} responses")
        
        # Standardize the data
        self.scaler = StandardScaler()
        self.standardized_responses = self.scaler.fit_transform(historical_responses)
        
        # Determine optimal number of factors if not specified
        if n_latent_factors is None:
            n_latent_factors = self._estimate_optimal_factors()
            if self.verbose:
                print(f"Automatically determined optimal number of factors: {n_latent_factors}")
            
        # Fit factor analysis model
        self.fa = FactorAnalysis(n_components=n_latent_factors, random_state=42)
        self.fa.fit(self.standardized_responses)
        
        # Calculate polychoric correlation matrix
        self.polychoric_corr = self._calculate_polychoric_correlation()
        
        # Initialize information gain graph
        self.info_graph = self._build_information_graph()
        
        if self.verbose:
            print(f"Initialization complete. Information graph has {len(self.info_graph.nodes)} nodes and {len(self.info_graph.edges)} edges")
            
    def visualize_factor_loadings(self):
        """
        Visualize factor loadings as a heatmap
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.fa.components_, 
                   xticklabels=self.question_labels,
                   yticklabels=[f"Factor {i}" for i in range(self.fa.components_.shape[0])],
                   cmap='RdBu_r', center=0, annot=True, fmt='.2f')
        plt.title('Factor Loadings Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def visualize_correlation_matrix(self):
        """
        Visualize polychoric correlation matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.polychoric_corr, 
                   xticklabels=self.question_labels,
                   yticklabels=self.question_labels,
                   cmap='RdBu_r', center=0, annot=True, fmt='.2f')
        plt.title('Question Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def visualize_information_graph(self):
        """
        Visualize the information graph structure
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.info_graph)
        
        # Draw question nodes
        question_nodes = [n for n in self.info_graph.nodes if n.startswith('Q')]
        nx.draw_networkx_nodes(self.info_graph, pos, nodelist=question_nodes,
                             node_color='lightblue', node_size=500)
        
        # Draw factor nodes
        factor_nodes = [n for n in self.info_graph.nodes if n.startswith('F')]
        nx.draw_networkx_nodes(self.info_graph, pos, nodelist=factor_nodes,
                             node_color='lightgreen', node_size=700)
        
        # Draw edges with weights
        edges = self.info_graph.edges(data=True)
        weights = [e[2]['weight'] * 2 for e in edges]
        nx.draw_networkx_edges(self.info_graph, pos, width=weights, alpha=0.5)
        
        # Add labels
        labels = {n: n for n in self.info_graph.nodes}
        nx.draw_networkx_labels(self.info_graph, pos, labels)
        
        plt.title('Information Flow Graph')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def _estimate_optimal_factors(self) -> int:
        """
        Estimate optimal number of latent factors using parallel analysis
        """
        if self.verbose:
            print("\nEstimating optimal number of factors...")
            
        n_datasets = 100
        max_factors = min(self.n_questions - 1, 10)
        real_data = np.zeros(max_factors)
        random_data = np.zeros(max_factors)
        
        real_evals = np.linalg.eigvals(np.cov(self.standardized_responses.T))
        real_evals.sort()
        real_evals = real_evals[::-1]
        
        for i in range(n_datasets):
            random_resp = np.random.normal(size=self.standardized_responses.shape)
            random_evals = np.linalg.eigvals(np.cov(random_resp.T))
            random_evals.sort()
            random_evals = random_evals[::-1]
            # Truncate to max_factors
            random_evals = random_evals[:max_factors]
            random_data += random_evals     


        random_data /= n_datasets
        
        # Visualize parallel analysis
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_factors + 1), real_evals[:max_factors], 
                'b-', label='Actual Data')
        plt.plot(range(1, max_factors + 1), random_data[:max_factors], 
                'r--', label='Random Data')
        plt.xlabel('Factor Number')
        plt.ylabel('Eigenvalue')
        plt.title('Parallel Analysis Scree Plot')
        plt.legend()
        plt.show()
        
        # Find optimal number
        for i in range(max_factors):
            if real_evals[i] < random_data[i]:
                if self.verbose:
                    print(f"Optimal number of factors: {i}")
                return i
                
        return max_factors
    
    def _calculate_polychoric_correlation(self) -> np.ndarray:
        """
        Calculate polychoric correlation matrix for ordinal data
        """
        corr_matrix = np.zeros((self.n_questions, self.n_questions))
        
        for i in range(self.n_questions):
            for j in range(i + 1):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Calculate polychoric correlation
                    try:
                        corr, _ = stats.spearmanr(
                            self.responses[:, i], 
                            self.responses[:, j]
                        )
                    except:
                        corr = 0.0
                    
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    
        return corr_matrix
    
    def _build_information_graph(self) -> nx.Graph:
        """
        Build weighted graph representing information relationships between questions
        """
        G = nx.Graph()
        
        # Add nodes for questions and latent factors
        for i in range(self.n_questions):
            G.add_node(f'Q{i}', type='question')
            
        for i in range(self.fa.components_.shape[0]):
            G.add_node(f'F{i}', type='factor')
            
        # Add edges between questions and factors
        for i in range(self.n_questions):
            for j in range(self.fa.components_.shape[0]):
                weight = abs(self.fa.components_[j, i])
                if weight > 0.1:  # Threshold to avoid weak connections
                    G.add_edge(f'Q{i}', f'F{j}', weight=weight)
                    
        return G

    def calculate_residual_entropy(self, 
                                 question_idx: int, 
                                 answered_questions: List[int]) -> float:
        """
        Calculate residual entropy for a question given already answered questions
        """
        if not answered_questions:
            return 1.0
            
        factor_loadings = self.fa.components_[:, question_idx]
        explained_variance = 0.0
        
        for i, loading in enumerate(factor_loadings):
            factor_determination = max([abs(self.fa.components_[i, q]) 
                                     for q in answered_questions], default=0)
            explained_variance += loading**2 * (1 - factor_determination)
            
        residual_entropy = -np.log2(1 - min(explained_variance, 0.9999))
        
        if self.verbose:
            print(f"\nResidual entropy for {self.question_labels[question_idx]}: {residual_entropy:.3f}")
            print(f"Explained variance: {explained_variance:.3f}")
            
        return residual_entropy

    def calculate_information_gain(self, 
                                 question_idx: int,
                                 answered_questions: List[int]) -> float:
        """
        Calculate expected information gain from asking a question
        """
        if self.verbose:
            print(f"\nCalculating information gain for {self.question_labels[question_idx]}")
            
        if not answered_questions:
            gain = stats.entropy(np.histogram(
                self.responses[:, question_idx], 
                bins='auto')[0]
            )
            if self.verbose:
                print(f"First question - using marginal entropy: {gain:.3f}")
            return gain
            
        info_value = 0.0
        
        # Direct information gain
        residual_entropy = self.calculate_residual_entropy(
            question_idx, answered_questions)
        info_value += residual_entropy
        
        if self.verbose:
            print(f"Direct information gain: {residual_entropy:.3f}")
        
        # Indirect information gain
        question_node = f'Q{question_idx}'
        indirect_gain = 0.0
        
        for factor in nx.neighbors(self.info_graph, question_node):
            factor_weight = self.info_graph[question_node][factor]['weight']
            
            for other_q in range(self.n_questions):
                if (other_q != question_idx and 
                    other_q not in answered_questions and
                    f'Q{other_q}' in self.info_graph[factor]):
                    
                    other_weight = self.info_graph[f'Q{other_q}'][factor]['weight']
                    contribution = (factor_weight * other_weight * 
                                  self.calculate_residual_entropy(other_q, answered_questions))
                    indirect_gain += contribution
                    
        info_value += indirect_gain
        
        if self.verbose:
            print(f"Indirect information gain: {indirect_gain:.3f}")
            print(f"Total information gain: {info_value:.3f}")
            
        return info_value

    def select_next_question(self, answered_questions: List[int]) -> int:
        """
        Select the next question that maximizes information gain
        """
        available_questions = [i for i in range(self.n_questions) 
                             if i not in answered_questions]
        
        info_gains = {
            q: self.calculate_information_gain(q, answered_questions)
            for q in available_questions
        }
        
        return max(info_gains.items(), key=lambda x: x[1])[0]
    
    def order_questions(self) -> List[int]:
        """
        Generate optimal ordering of all questions
        """
        if self.verbose:
            print("\nOrdering questions...")
            
        ordered_questions = []
        answered_questions = []
        
        # Track information gain at each step
        cumulative_info = []
        
        while len(ordered_questions) < self.n_questions:
            next_q = self.select_next_question(answered_questions)
            ordered_questions.append(next_q)
            answered_questions.append(next_q)
            
            info_gain = self.calculate_information_gain(next_q, answered_questions[:-1])
            cumulative_info.append(info_gain)
            
            if self.verbose:
                print(f"\nSelected question {len(ordered_questions)}: {self.question_labels[next_q]}")
                print(f"Information gain: {info_gain:.3f}")
        
        # Visualize cumulative information gain
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_info) + 1), cumulative_info, 'b-o')
        plt.xlabel('Question Number')
        plt.ylabel('Information Gain')
        plt.title('Cumulative Information Gain')
        plt.grid(True)
        plt.show()
        
        if self.verbose:
            print("\nFinal question order:")
            for i, q in enumerate(ordered_questions, 1):
                print(f"{i}. {self.question_labels[q]}")
        
        return ordered_questions    