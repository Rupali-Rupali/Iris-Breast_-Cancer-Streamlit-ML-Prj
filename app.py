import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#st.header("jai shree shyam ")
st.header("Titanic Survival Prediction")

header=st.container()
datasets=st.container()
features=st.container()
model_traning=st.container()


with datasets:
    st.header("We will work on Titanic Dataste")
    df=sns.load_dataset('titanic')
    df=df.dropna()
    st.write(df.head(10))
    st.subheader("Male vs Female Distribution")
    st.bar_chart(df['sex'].value_counts())

    # other plots
    st.subheader("Class Wise Distribution")
    st.bar_chart(df['class'].value_counts())
    st.subheader("Survived  Vs Not Survived")
    st.bar_chart(df['survived'].value_counts())

     
with features:
    st.header("These are our app features ")
    l=[i for i in df.columns]
    st.write(l)

with model_traning:
    #st.header("These are our app features ")
    input,display=st.columns(2)
    max_depth=input.slider(" How many people do yu know ",min_value=10,max_value=100,value=20,step=5)
# nestimtors
n_estimators=input.selectbox("How many trees should be there in random Forest",options=[50,100,200,300,'No Limit']
                             )
input.write()
input_text=input.text_input("Which Feature you want to select")

model=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)

X=df[[input_text]]
y=df[["fare"]]
# Display metrics
model.fit(X,y)
y_pred=model.predict(y)
display.subheader("Mean absoulte error is")
display.write(mean_absolute_error(y,y_pred))
display.subheader("Mean Squared error is :")
display.write(mean_squared_error(y,y_pred))
display.subheader("r2_score is :")
display.write(r2_score(y,y_pred))