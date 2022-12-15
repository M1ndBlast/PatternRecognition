import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from functools import reduce


np.set_printoptions(precision=2)

def report_average(reports:list):
    report_list = list()
    for report in reports:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        header = [x for x in splited[0].split(' ')]
        data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
        data = np.delete(data, 0, 1).astype(float)
        avg_total = np.array([x for x in splited[2].split(' ')][3:]).astype(float).reshape(-1, len(header))
        df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
        report_list.append(df)
    res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)
    return res.rename(index={res.index[-1]: 'avg / total'})


def classify(estimator, X:np.ndarray, y:np.ndarray, labels, fit_fun=None, predict_fun=None, score_fun=None, title:str="", n_splits:int=2, test_size=0.2, by_iterations:bool=False):
	# kf = KFold(n_splits=(n_splits if n_splits>=2 else 2), shuffle=True, random_state=0)
	kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
	
	# if not score_fun:
	# 	score_fun = estimator.score
	# 	# print("Using estimator score")
	# if not predict_fun:
	# 	predict_fun = estimator.predict
	# 	# print("Using estimator predict")

	if not by_iterations:
		scores = cross_validate(estimator, X, y, cv=kf, return_train_score=True, return_estimator=True)
	else:
		scores = {}
		scores["train_score"] = list()
		scores["test_score"] = list()
		scores["fit_time"] = list()
		scores["score_time"] = list()
		scores["report"] = list()
		
		for train_index, test_index in kf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			# fit
			print(f"Train: {len(X_train)} - Test: {len(X_test)}")
			fit_start = time()
			if fit_fun:
				fit_fun(X_train, y_train)
			else:
				estimator.fit(X_train, y_train)
			fit_end = time()
			print(f"Fit Time: {fit_end - fit_start}ms")
			
			# score train
			scores["train_score"].append(score_fun(X_train, y_train) if score_fun else estimator.score(X_train, y_train))
			print(f"Train Score: {scores['train_score'][-1]*100}%")

			# score test
			score_start = time()
			scores["test_score"].append(score_fun(X_test, y_test) if score_fun else estimator.score(X_test, y_test))
			score_end = time()
			print(f"Test Score: {scores['test_score'][-1]*100}%")

			# classification report
			
			y_pred = predict_fun(X_test) if predict_fun else estimator.predict(X_test)
			print("Predicted: ", len(y_pred))
			scores["report"].append(classification_report(y_test, y_pred, labels=labels, output_dict=True))
			print("report end")

			# fit time
			scores["fit_time"].append(fit_end - fit_start)
			# score time
			scores["score_time"].append(score_end - score_start)
		

	print(f"\t - {title} -")
	print(f"\t\t - Train Scores: {scores['train_score']*100}%")
	print(f"\t\t - Scores: {scores['test_score']*100}%")
	print(f"\t\t - Mean: {scores['test_score'].mean()*100}%")
	print(f"\t\t - Std: {scores['test_score'].std()*100}%")
	print(f"\t\t - Fit Time: {scores['fit_time'].mean()*1000}ms")
	print(f"\t\t - Score Time: {scores['score_time'].mean()*1000}ms")
	# Print classification report if exists
	if "report" in scores:
		print(report_average(scores["report"]))

	# plot scores
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel("Fold")
	ax.set_ylabel("RMSE")
	ax.plot(np.sqrt(scores["train_score"]), "o", label="Train")
	ax.plot(np.sqrt(scores["test_score"]), "x", label="Test")
	ax.legend()
	plt.show()


# makes regression
def regression(estimator, X:np.ndarray, y:np.ndarray, title:str="", test_size:float=0.2):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

	# fit
	fit_start = time()
	estimator.fit(X_train, y_train)
	fit_end = time()

	# score train
	train_score = estimator.score(X_train, y_train)

	# score test
	score_start = time()
	test_score = estimator.score(X_test, y_test)
	score_end = time()

	# fit time
	fit_time = fit_end - fit_start
	# score time
	score_time = score_end - score_start

	print(f"\t - {title} -")
	print(f"\t\t - Train Score: {train_score*100}%")
	print(f"\t\t - Score: {test_score*100}%")
	print(f"\t\t - Fit Time: {fit_time}ms")
	print(f"\t\t - Score Time: {score_time}ms")

	# plot real vs predicted
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel("Real")
	ax.set_ylabel("Predicted")
	y_pred = estimator.predict(X)
	ax.scatter(y, y_pred)
	plt.show()

	# plot scores
	fig, ax = plt.subplots()
	ax.set_title(title)
	ax.set_xlabel("Fold")
	ax.set_ylabel("RMSE")
	ax.plot(np.sqrt(train_score), "o", label="Train")
	ax.plot(np.sqrt(test_score), "o", label="Test")
	ax.legend()
	plt.show()