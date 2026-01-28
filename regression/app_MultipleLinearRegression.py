import streamlit as st
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

# page config #
st.set_page_config("MULTI LINEAR REGRESSION",layout="centered")

#load css
def load_css(file):
    with open (file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")


#title

st.markdown("""
            <div class="card">
            <h1>Multi Linear Regression</h1>
            <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Multi Linear Regression</p>
            </div>
           """ ,unsafe_allow_html=True)

#Load data

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df=load_data()

#Dataset Preview 

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["total_bill","size","tip"]].head())
st.markdown('</div>',unsafe_allow_html=True)

#Prepare Data

x,y=df[["total_bill","size"]],df["tip"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#Train model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

#Metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2) * (len(y_test)-1)/(len(y_test)-2)

#Visualization
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#Performance
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance")
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R2",f"{r2:.3f}")
c4.metric("adj r2",f"{adj_r2: .3f}")
st.markdown('</div>',unsafe_allow_html=True)

#coefficeints
st.markdown(f""" 
            <div class="card">
            <h3>Model Interception</h3>
            <p><b>co-efficient(total bill):</b>{model.coef_[0]: .3f}<br>
            <b>co-efficient(Group size):</b>{model.coef_[1]: .3f}<br>
            <b>Intercept:</b>{model.intercept_:.3f}
            </p>
            </div>
            """,unsafe_allow_html=True)

 #Predicitions
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predicition Tip for Total Bill")
bill = st.slider(
    "Select Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)
s=st.slider(
    "Group size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)
tip = model.predict(scaler.transform([[bill,s]]))[0]
st.markdown(
    f'<div class="prediction-box">Predicted Tip: {tip:.2f}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
