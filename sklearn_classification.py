# KNN 3
# KNN 5
# Naive Bayes
# NN
# Decision Tree
# SVM linear
# SVM poly
# SVM sigmoid

import os, re, time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer

from util import classify

DIR = "Classification"
DATASET_FILENAME = DIR+os.sep+"Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx"
CHECKPOINT_FILENAME_1 = DIR+os.sep+"lemmas_adj.npy"
CHECKPOINT_FILENAME_2_x = DIR+os.sep+"Rest_Mex_Preprocessed-x.npy"
CHECKPOINT_FILENAME_2_y = DIR+os.sep+"Rest_Mex_Preprocessed-y.npy"

CROSS_VALIDATION_TIMES = 2	#5


"""	Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx 

	Samples: #30,213
	Features: #2
	Type of Classifications: #2	

	Structure:
		- Title			-> str
		- Opinion		-> str
		- Polarity		-> int {1,2,3,4,5}	
							1 - Very Negative
							2 - Negative
							3 - Neutral
							4 - Positive
							5 - Very Positive
		- Attraction	-> str {Hotel, Restaurant, Attractive}
"""
target_names = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

def dataset_resume(X:np.ndarray, y:np.ndarray, title:str="Dataset Resume"):
	target_names, target_counts = np.unique(y, return_counts=True)
	count_tot = target_counts.sum()

	y_stats = pd.DataFrame(data={
		"Name":target_names,
		"Count":target_counts, 
		"Porcent":np.array([c*100/count_tot for c in target_counts])
	})

	print("\n")
	print("*"*(len(title)+32))
	print(f"{'*'*15} {title} {'*'*15}")
	print(f"\n\t - Stadistics of Classification -")
	print(f"{y_stats}")
	print(f"\n\t - Data -")
	print(X)
	print("*"*(len(title)+32))
	print("\n")


def oversapling(X:np.ndarray, y:np.ndarray):
	from imblearn.over_sampling import RandomOverSampler

	X, y = RandomOverSampler(random_state=0).fit_resample(X,y)
	return X, np.reshape(y, (y.shape[0],1))


def pln_preprocessed(X:np.ndarray):
	if not os.path.exists(CHECKPOINT_FILENAME_1):	
		import spacy
		nlp = spacy.load("es_core_news_lg")
		
		X_preprocessed = []
		i = 1
		for i, row in enumerate(X.values):
			print(f"\rPreprocessing {i}/{len(X.values)}", end="")
			text = ". ".join([re.sub(r"[\t ]+", " ", re.sub(r"(^\s|\s$)", "", s, 0, re.MULTILINE), 0, re.MULTILINE) for s in row])
			doc = nlp(text)
			lemmas = " ".join([token.lemma_ for token in doc if token.pos_ in ["ADJ"]])
			X_preprocessed.append(lemmas)
		X_preprocessed = np.array(X_preprocessed)

		np.save(CHECKPOINT_FILENAME_1, X_preprocessed)
		print(f"CHECKPOINT '{CHECKPOINT_FILENAME_1}' SAVED")
	else:
		X_preprocessed = np.load(CHECKPOINT_FILENAME_1)
		print(f"CHECKPOINT '{CHECKPOINT_FILENAME_1}' LOADED")

	freq_count_vectorizer = CountVectorizer()
	X_preprocessed = freq_count_vectorizer.fit_transform(X_preprocessed).toarray()
	vocabulary = freq_count_vectorizer.get_feature_names_out()

	return X_preprocessed, vocabulary


def getNN(
	hidden_layers:tuple, 
	learning_rate:float=0.1, 
	in_num:int=100, 
	out_num:int=5, 
	activation_hidden:nn.Module=nn.ReLU(), 
	activation_output:nn.Module=nn.Sigmoid(), 
	backpropagation_fun:optim.Optimizer=optim.Adam
):
	layers = []
	for i in range(0, hidden_layers[0]+1):
		activation = activation_hidden
		if i==hidden_layers[0]:
			activation = activation_output

		print(f'({in_num if i==0 else hidden_layers[1]},{out_num if i==hidden_layers[0] else hidden_layers[1]})', end='\n' if i==hidden_layers[0] else ' -> ')

		layer = nn.Linear(
			in_num if i==0 else hidden_layers[1], 
			out_num if i==hidden_layers[0] else hidden_layers[1],
			False
		)
		
		layers.append((f'layer{i}', layer))
		layers.append((f'fun{i}', activation))

	from collections import OrderedDict
	model = nn.Sequential(OrderedDict(layers))
	optimizer = backpropagation_fun(model.parameters(), lr=learning_rate)

	return model, optimizer


if __name__=="__main__":
	### BEGIN DATASET LOADING ###
	if os.path.exists(CHECKPOINT_FILENAME_2_x):
		restmex_X = np.load(CHECKPOINT_FILENAME_2_x)
		restmex_Y = np.load(CHECKPOINT_FILENAME_2_y)
		print(f"CHECKPOINT '{CHECKPOINT_FILENAME_2_x}' LOADED")
	else:
	#	Rest Mex Dataset
		restmex_dataset = pd.read_excel(DATASET_FILENAME, dtype=str)\
			.replace(to_replace=np.NaN,value="")

		restmex_X = np.array([". ".join(patter) for patter in restmex_dataset.drop(["Polarity", "Attraction"], axis=1).values.tolist()])
		restmex_Y = restmex_dataset.drop(["Title", "Opinion", "Attraction"], axis=1)\
			.astype(int).values
		
	#	PLN Preprocessing
		restmex_X, vocabulary = pln_preprocessed(restmex_X)
		dataset_resume(restmex_X, restmex_Y, "Rest Mex Sentiment Analysis Dataset")

	#	OverSampling Preprocessing
		restmex_X, restmex_Y = oversapling(restmex_X, restmex_Y)
		#dataset_resume(restmex_X, restmex_Y, "Rest Mex Sentiment Analysis Dataset W/ Oversampling")

		np.save(CHECKPOINT_FILENAME_2_x, restmex_X)
		np.save(CHECKPOINT_FILENAME_2_y, restmex_Y)
		print(f"CHECKPOINT '{CHECKPOINT_FILENAME_2_x}' SAVED")
	### END OF DATASET LOADING ###
	dataset_resume(restmex_X, restmex_Y, "Rest Mex Sentiment Analysis Dataset W/ Oversampling")
	restmex_Y = np.reshape(restmex_Y, (restmex_Y.shape[0],))

	### BEGIN CLASSIFICATION ###
	print("CLASSIFICATION")

#	KNN
	#	-	1NN
	knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
	classify(knn, restmex_X, restmex_Y, labels=target_names, title="KNN - 1NN", by_iterations=True)

	#	-	3NN
	knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
	classify(knn, restmex_X, restmex_Y, labels=target_names, title="KNN - 3NN", by_iterations=True)

	#	-	5NN
	knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
	classify(knn, restmex_X, restmex_Y, labels=target_names, title="KNN - 5NN", by_iterations=True)


#	Naive Bayes
	gnb = GaussianNB()
	classify(gnb, restmex_X, restmex_Y, labels=target_names, title="Naive Bayes", by_iterations=True)
	

#	Decision Tree
	dt = DecisionTreeClassifier(random_state=0)
	classify(dt, restmex_X, restmex_Y, labels=target_names, title="Decision Tree", by_iterations=True)


#	SVM
	#	-	Linear Kernel
	svm = SVC(kernel="linear", C=1)
	classify(svm, restmex_X, restmex_Y, labels=target_names, title="SVM - Linear Kernel", by_iterations=True)

	#	-	Polynomial Kernel
	svm = SVC(kernel="poly", C=1)
	classify(svm, restmex_X, restmex_Y, labels=target_names, title="SVM - Polynomial Kernel", by_iterations=True)

	#	-	Sigmoid Kernel
	svm = SVC(kernel="sigmoid", C=1)
	classify(svm, restmex_X, restmex_Y, labels=target_names, title="SVM - Sigmoid Kernel", by_iterations=True)

# 	Neural Network
	import torch as t
	from torch import nn, optim
	restmex_Y = np.reshape(restmex_Y, (restmex_Y.shape[0],))
	restmex_X = t.FloatTensor(restmex_X)
	restmex_Y = t.LongTensor(restmex_Y)-1

	nnClassifier, opt = getNN(
		hidden_layers = (2, restmex_X.shape[1]//3),
		learning_rate=0.1,
		in_num = restmex_X.shape[1],
		out_num = 5,
		activation_hidden = nn.ReLU(),
		activation_output = nn.Softmax(dim=1), #lambda x: t.heaviside(x, t.tensor([0.]))
		backpropagation_fun = optim.SGD
	)

	def nn_fit(X, y):
		time_batch_start = time()
		time_batch_end = time()
		epochs = 50 # 8 mins per epoch
		bs = 512
		n = X.shape[0]
		criterion = nn.CrossEntropyLoss(reduction='mean')
		loss = t.tensor([[0]])
		
		for epoch in range(epochs):
			for i in range((n-1)//bs+1):
				print(f'\rLoss {loss:.4f} Epoch {epoch+1}/{epochs} Batch {i+1}/{(n-1)//bs+1} ({(time_batch_end-time_batch_start):.2f} seconds)', end=' '*6)
				time_batch_start = time()

				opt.zero_grad()

				start_i = i * bs
				end_i = start_i + bs
				xb = X[start_i:end_i]
				yb = y[start_i:end_i]

				yb_probs = nnClassifier.forward(xb)

				loss = criterion(yb_probs, yb)
				loss.backward()
				opt.step()

				time_batch_end = time()
			# if epoch%2==0:
			# 	view_classify(yb_probs[0], yb[0])

	def nn_predict(X):
		with t.no_grad():
			y_probs = nnClassifier.forward(X)
			y_pred = t.argmax(y_probs, dim=1)
			return y_pred

	def nn_score(X, y):
		from sklearn.metrics import accuracy_score
		y_pred = nn_predict(X)
		return accuracy_score(y, y_pred)

	classify(None, restmex_X, restmex_Y, labels=target_names, title="Neural Network", by_iterations=True, n_splits=1, fit_fun=nn_fit, predict_fun=nn_predict, score_fun=nn_score)


	### END OF CLASSIFICATION ###
