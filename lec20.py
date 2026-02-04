import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

st.title("APP")
uploader=st.file_uploader("select data",type=["csv"])
if uploader is not None:
    df=pd.read_csv(uploader)
    st.write("Header")
    st.table(df.head())
    if st.checkbox("dataset"):
        st.write("Description of the file")
        st.table(df.describe())
        
    if st.checkbox("missing"):
        st.write(df.isna().sum())
        if st.button("Delete Missing DATA"):
            df.dropna(inplace=True)
            st.success("Done")
            
    if st.checkbox("Duplications"):
        st.write(df.duplicated().sum())
        if st.button("delete duplications"):
            df.drop_duplicates(inplace=True)
            st.success("Done")
            
    if st.checkbox("outliers"):
        fig,ax=plt.subplots()
        sns.boxplot(df,ax=ax)
        st.pyplot(fig)
        
    if st.checkbox("transformation"):
        cat=df.select_dtypes("object").columns
        st.write(cat)
        
        if cat is not None:
            st.warning("NO categorical")
        else:
            o=st.selectbox("select encoding way",options=["LabelEncoder","OneHotEncoder","OrdinalEncoder"])
            if o=="LabelEncoder":
                pass
            elif o=="OneHotEncoder":
                pass
            elif o=="OrdinalEncoder":
                pass
            
    
        
        
    if st.checkbox("splitting"):
        target=st.selectbox(options=df.columns)
        st.write(target)
        features=st.multiselect(options=df.columns)
        st.write(features)
        
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,shuffle=True,stratify=y)
        st.success("split")
        
        # st.header("module")
        # option=st.selectbox(options=["knn","lr"])
        # if option=="knn":
        #     k=st.slider("n_neighbour",2,20,10)
        #     weights=st.selectbox(options=["uniform"])
        #     metric=st.selectbox(options=["manhattan"])
            