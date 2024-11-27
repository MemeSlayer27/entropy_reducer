from entropy_reducer import EnhancedQuestionOrderer
import pandas as pd
import numpy as np



data = pd.read_csv('data.csv')
data = data.iloc[:, [1]].join(data.iloc[:, 3:32])

data.replace('-', np.nan, inplace=True)
nan_threshold = (len(data.columns) - 1) * 0.8
data = data.dropna(thresh=nan_threshold, subset=data.columns[1:])
data.fillna(3, inplace=True)

print(data.head())


answer_vectors = data.iloc[:, 1:].values

# turn str values into int
responses = answer_vectors.astype(int)

print(responses)

labels = data.columns[1:]

# turn labels into list
labels = list(labels)
print(labels)

# Initialize orderer
orderer = EnhancedQuestionOrderer(responses, question_labels=labels, verbose=True)

# Visualize different aspects
orderer.visualize_factor_loadings()
orderer.visualize_correlation_matrix()
orderer.visualize_information_graph()

# Get ordered questions
ordered_questions = orderer.order_questions()

