import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
st.header("App making using Plotly and streamlit ")
df=px.data.gapminder()
st.write(df)
st.write(df.columns)
st.write(df.describe())
year_option=df["year"].unique().tolist()
year=st.selectbox("Which year should we plot",year_option)
df=df[df['year']==year]
fig=px.scatter(df,x="gdpPercap",y="lifeExp",size="pop",hover_name="continent",color="continent",
           log_x=True,size_max=55,range_x=[100,10000],range_y=[20,90],animation_frame="year",animation_group="country")
fig.update_layout(width=800)
st.write(fig)
