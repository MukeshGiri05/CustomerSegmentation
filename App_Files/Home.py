import streamlit as st
import pandas as pd

def Home_Page():
    # Display a centered, styled heading
    st.markdown("<h2 style='text-align: center; color: red;'>Customer Segmentation and Purchase Prediction System</h2>", unsafe_allow_html=True)

    # Display the welcome message
    st.write("""
    Welcome to our innovative Customer Segmentation and Future Product Purchase Prediction System. This cutting-edge web application is designed to revolutionize the way businesses understand and engage with their customers. By leveraging advanced analytics and machine learning techniques, our system offers a comprehensive approach to customer segmentation and future purchase predictions.
    """)

    # Section 1: Multi-Attribute Segmentation
    ##Wrong discription
    st.header("1. Multi-Attribute Segmentation")
    st.write("""
    Our system segments customers based on a wide range of attributes, including demographics, behavioral patterns, and transactional data. This allows businesses to tailor their marketing and product development strategies to meet the unique needs and preferences of different customer segments.
    """)
    st.subheader("Data used for Multi-Attribute Segmentation")
    st.dataframe(pd.read_csv('App_Data/Customer_Data.csv'))

    # Section 2: RFM Analysis
    st.header("2. RFM Analysis")

    ##Wrong discription
    st.write("""
    Recency, Frequency, and Monetary (RFM) analysis is a powerful tool for understanding customer behavior. Our system performs RFM analysis to identify customers who are most likely to churn, enabling businesses to take proactive measures to retain these valuable customers.
    """)
    st.subheader("Data used for RFM Analysis")
    st.dataframe(pd.read_csv('App_Data/Invoice_Data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str}))

    # Section 3: Churn Prediction
    st.header("3. Churn Prediction")
    st.write("""
    By analyzing historical data and customer behavior, our system predicts which customers are at risk of churning. This predictive analytics capability enables businesses to intervene early and offer personalized retention strategies, significantly reducing customer attrition.
    """)
    st.subheader("Data used for Churn Prediction")
    st.dataframe(pd.read_csv('App_Data/data_for_churn.csv'))

    # Section 4: Future Product Purchase Prediction
    st.header("4. Future Product Purchase Prediction")

    ##tell about market basket analysis
    st.write("""
    After segmenting customers and analyzing their behavior, our system uses machine learning models to predict which products a customer is most likely to purchase in the future. This feature empowers businesses to offer personalized product recommendations, enhancing customer satisfaction and increasing sales.
    """)
    st.subheader("Data used for product analysis and purchase prediction")
    st.dataframe(pd.read_csv('App_Data/Invoice_Data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str}))

    ##Add about visualization module

    
