import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
from App_Files import Constants as cn

def user_input(prompt):
    uploaded_file = st.file_uploader(prompt, type='csv')
    if uploaded_file != None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        pass
    
@st.cache_data
def churn_pred(df,modelpath):
    df.set_index('customerID',inplace=True)
    st.subheader("Data Overview")
    st.dataframe(df)

    df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]
    df1 = df[df.TotalCharges!=' ']
    df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

    df1.replace('No internet service','No',inplace=True)
    df1.replace('No phone service','No',inplace=True)

    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yes_no_columns:
        df1[col].replace({'Yes': 1,'No': 0},inplace=True)


    df1['gender'].replace({'Female':1,'Male':0},inplace=True)

    ## One hot Encoding
    df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod']).astype(int)

    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

    model = load_model(modelpath)


    prediction = model.predict(df2)


    prediction

    pred = []
    for element in prediction:
            if element > 0.5:
                pred.append(1)
            else:
                pred.append(0)

    dic = {"customerID":df2.index.values,"Churn":pred}
    result = pd.DataFrame(dic)

    result['Churn'] = result['Churn'].replace({0: 'No'})
    result['Churn'] = result['Churn'].replace({1: 'Yes'})


    st.subheader("Churn Prediction Result")
    st.dataframe(result)

    result.rename(columns={'customerID': 'customerID'}, inplace=True)

    df1.rename(columns={'customerID': 'customerID'}, inplace=True)

    df1 = df1.reset_index().rename(columns={'index': 'customerID'})

    result = result.merge(df1[['customerID', 'tenure', 'MonthlyCharges']], on='customerID', how='left')

    ## Plotting Graphs
    tenure_churn_no = result[result.Churn=='No'].tenure
    tenure_churn_yes = result[result.Churn=='Yes'].tenure

    ## Plotting Graph
    plt.xlabel("Time Period")
    plt.ylabel("Number Of Customers")
    plt.title("Customer Churn Prediction Visualiztion")

    plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
    plt.legend()
    st.subheader("Histogram of Time Period")
    st.pyplot(plt)
    plt.clf()

    mc_churn_no = result[result.Churn=='No'].MonthlyCharges      
    mc_churn_yes = result[result.Churn=='Yes'].MonthlyCharges      

    plt.xlabel("Monthly Expenses")
    plt.ylabel("Number Of Customers")
    plt.title("Customer Churn Prediction Visualiztion")

    plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
    plt.legend()
    st.subheader("Histogram of Monthly Expenses")
    st.pyplot(plt)
    plt.clf()

    ##Plotting Ends

    Not_Churned = result[result['Churn']=='No']
    Churned = result[result['Churn']=='Yes']
    with st.expander("Customers who did not churn"):
        st.dataframe(Not_Churned)

    with st.expander("Customers Who Churned"):
        st.dataframe(Churned)
 
    dicto = {"Labels":"Count",
             "Not Churned":Not_Churned['customerID'].count(),
            "Churned":Churned['customerID'].count()}
    
    st.subheader("Summary")
    st.dataframe(dicto)

def run():
    data = user_input("Enter data for Churn Prediction")
    modelpath = cn.ChurnPredictionModelPath

    if data is not None:
        with st.spinner():
            churn_pred(data,modelpath)
    else:
        st.success("Enter Customer Purchase Data For Churn Prediction")