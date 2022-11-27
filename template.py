#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_dataset(dataset_path):
	data=pd.read_csv(dataset_path)
	return data

def dataset_stat(dataset_df):	
	n_feats = dataset_df.shape[1]
	n_class0=dataset_df[dataset_df['target']==0].shape[0]
	n_class1=dataset_df[dataset_df['target']==1].shape[0]
	return n_feats,n_class0,n_class1

def split_dataset(dataset_df, testset_size):
	x=dataset_df.loc[:,dataset_df.columns!='target']
	y=dataset_df.loc[:,['target']]
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testset_size)
	return x_train,x_test,y_train,y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt=DecisionTreeClassifier(max_depth=2,random_state=1)
	dt=dt.fit(x_train,y_train)
	y_pred=dt.predict(x_test)
	accuracy=format(accuracy_score(y_test,y_pred))
	precision=format(precision_score(y_test,y_pred))
	recall=format(recall_score(y_test,y_pred))
	return accuracy,precision,recall
    
def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf=RandomForestClassifier(random_state=200)
	rf.fit(x_train,y_train.values.ravel())
	y_pred=rf.predict(x_test)
	accuracy=format(accuracy_score(y_test,y_pred))
	precision=format(precision_score(y_test,y_pred))
	recall=format(recall_score(y_test,y_pred))
	return accuracy,precision,recall

def svm_train_test(x_train, x_test, y_train, y_test):
	steps=[('scaler',StandardScaler()),('SVM',svm.SVC())]
	pipeline=Pipeline(steps)
	pipeline.fit(x_train,y_train.values.ravel())
	y_pred=pipeline.predict(x_test)
	accuracy=format(accuracy_score(y_test,y_pred))
	precision=format(precision_score(y_test,y_pred))
	recall=format(recall_score(y_test,y_pred))
	return accuracy,precision,recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)