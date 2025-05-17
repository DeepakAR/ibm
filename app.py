
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


def load_data(file_path):
    return pd.read_csv(file_path)

def check_null_values(df):
    return df.isnull().sum()

def check_duplictes(df):
    return df.duplicated().sum()

def detect_anomolies(df,columns):
    df_numeric=df[columns].select_dtypes(inlcude=[np.number])
    if df_numeric.empty:
        return pd.DataFrame
    
    df_numeric.fillna(df_numeric.median(),inplace=True)

    model=IsolationForest(contamination=0.05,random_state=42)
    df_numeric["anomaly"]=model.fit_predict(df_numeric)
    anomalies=df[df_numeric["anomaly"]==-1]

    return anomalies


st.title("AI based Data qualit monitering")

uploaded_file=st.file_uploader("upload CSV file",type=["csv","xlsx"])
if uploaded_file:
    file_path=uploaded_file.name
    data=load_data(uploaded_file)


    st.subheader("Data preview")
    st.dataframe(data.head())

    null_values=check_null_values(data)
    st.subheader("Missing values")
    st.write(null_values)


    st.subheader("Missing Values Visualisation")
    fig, ax=plt.subplots()
    sns.barplot(x=null_values.index,y=null_values.values,ax=ax,palette='coolwarm')
    ax.set_ylabel("count")
    ax.set_title("missing values per column")
    st.pyplot(fig)


    duplicates=check_duplictes(data)
    st.subheader("Duplicate rows")
    st.write(f"Total duplicates: {duplicates}")

    st.subheader("duplicate rows Distribution")
    fig,ax=plt.subplots()
    labels=["Unique Rows","Duplicates"]
    values=[len(data)-duplicates,duplicates]
    ax.pie(values,labels=labels,autopct="%1.1f%%",colors=["skyblue","red"])
    st.pyplot(fig)

    st.subheader("Select Columns for Anamoly Detection")
    selected_columns=st.multiselect("Choose Numeric Columns",data.select_dtypes(include=[np.number]).columns)

    if selected_columns:
        anamolies=detect_anomolies(data,selected_columns)
        st.subheader("detected anamolies")
        st.dataframe(anamolies)

        st.subheader("anomaly detection Visualisation")
        fig,ax=plt.subplots(figsize=(8,5))
        sns.boxplot(data=data[selected_columns],ax=ax,palette="Set2")
        ax.set_title("Box plot for outlies Detection")
        st.pyplot(fig)