import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
st.header("huj")
st.write("""
# Explore Different ML models and  Different Datasets
""")
dataset_name=st.sidebar.selectbox("Select Datasets ",
                                  ("Iris","Breast Cancer","Wine")
                                  )
classifier_name=st.sidebar.selectbox("Select Classifier",
                                  ("KVM","SVM","Random Forest")
                                  )
def get_dataset(dataset_name):
    data=None
    if dataset_name=='Iris':
        data=datasets.load_iris()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    else:
        data=datasets.load_breast_cancer()
    X=data.data
    y=data.target
    return X,y
X,y=get_dataset(dataset_name)
st.write("Shape of Dataset :",X.shape)
st.write("Number of classes ",len(np.unique(y)))
st.write("Name of classes ",np.unique(y))
def add_parameter(classifier_name):
    paramas=dict()
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        paramas['C']=C
    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,13)
        paramas['K']=K
    else:
        max_depth=st.sidebar.slider('max_depth',2,13)
        paramas['max_depth']=max_depth # depth of every trees that grow in rsndom forest
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        paramas['n_estimators']=n_estimators # num of trees
    return paramas

paramas=add_parameter(classifier_name)
def get_classifier(classifier_name,paramas):
    clf=None
    if classifier_name=="SVM":
        clf=SVC(C=paramas['C'])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=paramas['K'])
    else:
        clf=RandomForestClassifier(n_estimators=paramas['n_estimators'],
                                   max_depth=paramas['max_depth']
                                   )
    return clf
clf=get_classifier(classifier_name,paramas)  


# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier ={classifier_name}')

fig=plt.figure()
plt.scatter(y_test,y_pred,cmap='viridis')

plt.xlabel('Actual Value')

plt.colorbar()
plt.title("Actual Value Versus Predicted Value")
st.pyplot(fig)
st.write(f'Accuraccy Score is =',acc)