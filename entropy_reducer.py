import numpy as np
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

N_CATEGORIES = 5

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
        with correlation matrices and multiple criteria.
        
        Returns:
            int: Optimal number of factors
        """
        if self.verbose:
            print("\nEstimating optimal number of factors...")
            
        # Calculate correlation matrix for actual data
        corr_matrix = np.corrcoef(self.standardized_responses.T)
        real_evals = np.linalg.eigvals(corr_matrix)
        real_evals.sort()
        real_evals = real_evals[::-1]  # Sort in descending order
        
        # Generate random correlation matrices
        n_iterations = 100
        n_vars = self.standardized_responses.shape[1]
        max_factors = min(n_vars - 1, 10)  # Cap at 10 factors
        random_evals = np.zeros(max_factors)
        
        for _ in range(n_iterations):
            # Generate random normal data with same shape
            random_data = np.random.normal(size=self.standardized_responses.shape)
            # Get correlation matrix and eigenvalues
            random_corr = np.corrcoef(random_data.T)
            evals = np.linalg.eigvals(random_corr)
            evals.sort()
            random_evals += evals[::-1][:max_factors]
        
        # Average the random eigenvalues
        random_evals /= n_iterations
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_factors + 1), real_evals[:max_factors], 
                'b-', label='Actual Data')
        plt.plot(range(1, max_factors + 1), random_evals, 
                'r--', label='Random Data')
        plt.xlabel('Factor Number')
        plt.ylabel('Eigenvalue')
        plt.title('Parallel Analysis Scree Plot')
        plt.legend()
        plt.show()
        
        # Apply multiple criteria for factor selection:
        # 1. Kaiser criterion (eigenvalue > 1)
        # 2. Parallel analysis (actual > random)
        # 3. Proportion of variance explained
        
        cumulative_variance = np.cumsum(real_evals) / np.sum(real_evals)
        
        for i in range(max_factors):
            # Stop if eigenvalue is less than 1 (Kaiser criterion)
            if real_evals[i] < 1:
                return max(1, i)
            
            # Stop if eigenvalue is less than random data
            if real_evals[i] < random_evals[i]:
                return max(1, i)
            
            # Stop if we explain 80% of variance
            if cumulative_variance[i] > 0.8:
                return max(1, i + 1)
        
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

        residual_variance = max(1e-10, 1-explained_variance)
    
        # Base entropy from continuous approximation
        continuous_entropy = 0.5 * np.log(2 * np.pi * np.e * residual_variance)
        
        # Correction for discretization
        # Using mutual information between continuous and discretized variable
        width = np.sqrt(residual_variance) / N_CATEGORIES
        discretization_loss = -np.log(width)
        
        if self.verbose:
            print(f"\nResidual entropy for {self.question_labels[question_idx]}: {continuous_entropy:.3f}")
            print(f"Explained variance: {explained_variance:.3f}")

        # Additional correction for ordinal nature
        ordinal_correction = np.log(N_CATEGORIES)
        
        return max(0, continuous_entropy - discretization_loss + ordinal_correction)  

    def calculate_information_gain(self, 
                                 question_idx: int,
                                 answered_questions: List[int]) -> float:
        """
        Calculate expected information gain from asking a question
        """
        if self.verbose:
            print(f"\nCalculating information gain for {self.question_labels[question_idx]}")
            
        """if not answered_questions:
            gain = stats.entropy(np.histogram(
                self.responses[:, question_idx], 
                bins='auto')[0]
            )
            if self.verbose:
                print(f"First question - using marginal entropy: {gain:.3f}")
            return gain"""
            
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