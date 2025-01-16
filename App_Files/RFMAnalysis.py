import numpy as np 
import pandas as pd 
import warnings
import datetime as dt
from tensorflow.keras.models import load_model
import joblib
import streamlit as st
from App_Files import Constants as cn
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

def user_input(prompt):
    uploaded_file = st.file_uploader(prompt, type='csv')
    if uploaded_file != None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        pass

import tensorflow as tf
#model = tf.keras.models.load_model("App_Models////RFMSegmentationModel.keras")
model = tf.keras.models.load_model(r"C://Users//Mukes//Project//CustomerSegmentation//App_Models//RFMSegmentationModel.h5")


@st.cache_data
def rfm(retail_df,labelEncoderPath):
    st.subheader("Overview of Data")
    st.dataframe(retail_df.head())
    #remove canceled orders
    retail_uk = retail_df[retail_df['Quantity']>0]
    #remove rows where customerID are NA
    retail_uk.dropna(subset=['CustomerID'],how='all',inplace=True)
    #restrict the data to one full year because it's better to use a metric per Months or Years in RFM
    retail_uk = retail_uk[retail_uk['InvoiceDate']>= "2010-12-09"]

    ##First Result
    summary_stats = {
    "Number of transactions": retail_uk['InvoiceNo'].nunique(),
    "Number of products bought": retail_uk['StockCode'].nunique(),
    "Number of customers": retail_uk['CustomerID'].nunique(),
    "Percentage of customers NA": round(retail_uk['CustomerID'].isnull().sum() * 100 / len(retail_df),2)
    }
    summaryDF = pd.DataFrame([summary_stats])
    
    ##IMP

    st.subheader("Summary of Data")
    st.dataframe(summaryDF)
    ##IMP END

    ##RFM analysis Start
    now = dt.date(2011,12,9)
    #create a new column called date which contains the date of invoice only
    retail_uk['date'] = pd.DatetimeIndex(retail_uk['InvoiceDate']).date

    #Recency
    recency_df = retail_uk.groupby(by='CustomerID', as_index=False)['date'].max()
    recency_df.columns = ['CustomerID','LastPurshaceDate']
    #calculate recency
    recency_df['Recency'] = recency_df['LastPurshaceDate'].apply(lambda x: (now - x).days)
    #drop LastPurchaseDate as we don't need it anymore
    recency_df.drop('LastPurshaceDate',axis=1,inplace=True)
    #Recency Done

    #Frequency
    # drop duplicates
    retail_uk_copy = retail_uk
    retail_uk_copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
    #calculate frequency of purchases
    frequency_df = retail_uk_copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
    frequency_df.columns = ['CustomerID','Frequency']
    #Frequency Done

    #Monetary Value
    #create column total cost
    retail_uk['TotalCost'] = retail_uk['Quantity'] * retail_uk['UnitPrice']
    monetary_df = retail_uk.groupby(by='CustomerID',as_index=False).agg({'TotalCost': 'sum'})
    monetary_df.columns = ['CustomerID','Monetary']
    #Monetary done

    #Create RFM table
    temp_df = recency_df.merge(frequency_df,on='CustomerID')
    #merge with monetary dataframe to get a table with the 3 columns
    rfm_df = temp_df.merge(monetary_df,on='CustomerID')
    #use CustomerID as index
    rfm_df.set_index('CustomerID',inplace=True)
    temp = rfm_df.copy()

    #Model
    predictions = model.predict(temp)

    # Load the label encoder
    le = joblib.load(labelEncoderPath)

    # Get the index of the maximum probability
    max_prob_indices = np.argmax(predictions, axis=1)

    # Transform the indices back to original form
    original_labels = le.inverse_transform(max_prob_indices)

    temp['Labels'] =original_labels

    ##Second result

    st.subheader("RFM Table")
    st.dataframe(temp)
    
    st.subheader("Visualization of RFM values")
    RfmValuesVisualization(temp)

    ##Second Result
    ##Third IMP
    unique_counts = temp['Labels'].value_counts()
    unique_counts_df = unique_counts.reset_index()
    unique_counts_df.columns = ["Labels", "Count"]

    ##Third Result

    st.subheader("Count Of Labels")
    st.dataframe(unique_counts_df)

    ##End of Third Result

    ##Fourth Result
    rfmgraph(unique_counts_df)
    ##End of Fourth Result

    st.subheader("World map of Recency")
    st.image("App_Data/Images/World Map Recency.png")
    # merge_and_plot_world_map(retail_df,rfm_df)

    lostCustomers = temp[temp['Labels']=='Lost Customers']
    other = temp[temp['Labels']=='Other']
    BestCustomers = temp[temp['Labels']=='Best Customers']
    almostLost = temp[temp['Labels']=='Almost Lost']
    loyalCustomers = temp[temp['Labels']=='Loyal Customers']
    bigSpender = temp[temp['Labels']=='Big Spenders']

    c = st.container()
    c.header("RFM Labels Segmentation")
    c1,c2,c3,c4 = c.columns(4)

    # c1.subheader("List of Big Spenders")
    with c1.expander('List of Big Spenders'):
        st.dataframe(bigSpender)
    # c2.subheader("List of other customers")
    with c4.expander('List of other customers'):
        st.dataframe(other)
    # c3.subheader("List of Best Customers")
    with c3.expander('List of Best Customers'):
        st.dataframe(BestCustomers)
    # c4.subheader("List of Almost Lost Customers")
    with c2.expander('List of Almost Lost Customers'):
        st.dataframe(almostLost)

    c5,c6 = st.columns(2)
    # c5.subheader("List of Loyal Customers")
    with c5.expander('List of Loyal Customers'):
        st.dataframe(loyalCustomers)
    # c6.subheader("List of Lost Customers")
    with c6.expander("List of Lost Customers"):
        st.dataframe(lostCustomers)

def take_user_input(prompt):
    uploaded_file = st.file_uploader(prompt, type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
        return df
    else:
        pass
    return 

def rfmgraph(unique_counts_df):
    plt.bar(x=unique_counts_df['Labels'],height=unique_counts_df['Count'],color=['green','yellow','blue','violet','pink','purple'])
    plt.xticks(rotation=45)
    plt.title("Count Of Labels")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    st.subheader("Bar Graph of Label")
    st.pyplot(plt)
    plt.clf()
    plt.close()

def RfmValuesVisualization(df_rfm):
    colnames = ['Recency', 'Frequency', 'Monetary']
    bins_config = {
        'Recency': {'bins': range(50, 300, 20)}, # 0 to 20 with gaps of 5
        'Frequency': {'bins': range(0, 11, 1)}, # 1 to 10 with gaps of 2
        'Monetary': {'bins': range(0, 21, 1)} # Adjust as needed
    }
    
    for col in colnames:
        fig, ax = plt.subplots(figsize=(12,3))
        sns.histplot(df_rfm[col], bins=bins_config[col]['bins'], kde=False)
        ax.set_title('Distribution of %s' % col)
        st.pyplot(plt)
        plt.clf()
        plt.close()

    segments = ['Lost Customers', 'Almost Lost', 'Other', 'Best Customers', 'Big Spenders', 'Loyal Customers']

    for col in colnames:
        fig, ax = plt.subplots(figsize=(12,3))
        for segment in segments:
            # Plot the distribution for the current segment
            sns.histplot(df_rfm[df_rfm['Labels']==segment][col], label=segment, bins=bins_config[col]['bins'],kde=False)
        ax.set_title('Distribution of %s' % col)
        plt.legend()
        st.pyplot(plt)
        plt.clf()
        plt.close()

    agg_dict2 = {
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'sum'
    }

    # import squarify

    # df_analysis = df_rfm.groupby('Labels').agg(agg_dict2).sort_values(by='Recency').reset_index()
    # df_analysis.rename({'Labels': 'label', 'CustomerID': 'count'}, axis=1, inplace=True)
    # df_analysis['count_share'] = df_analysis['count'] / df_analysis['count'].sum()
    # df_analysis['monetary_share'] = df_analysis['Monetary'] / df_analysis['Monetary'].sum()
    # df_analysis['Monetary'] = df_analysis['Monetary'] / df_analysis['count']
    # colors = ['#37BEB0', '#DBF5F0', '#41729F', '#C3E0E5', '#0C6170', '#5885AF', '#E1C340', '#274472', '#F8EA8C', '#A4E5E0', '#1848A0']

    # for col in ['count', 'monetary']:
    #     labels = df_analysis['label'] + df_analysis[col + '_share'].apply(lambda x: ' ({0:.1f}%)'.format(x*100))

    #     fig, ax = plt.subplots(figsize=(16,6))
    #     squarify.plot(sizes=df_analysis[col], label=labels, alpha=.8, color=colors)
    #     ax.set_title('RFM Segments of Customers (%s)' % col)
    #     plt.axis('off')
    #     st.pyplot(plt)
    #     plt.clf()
    #     plt.close()

# def merge_and_plot_world_map(df_original, df_rfm):

#     # Merge the RFM data with the original data on 'CustomerID'
#     merged_data = df_original.merge(df_rfm, on='CustomerID')
    
#     # Load world map
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
#     # Merge the merged data with the world map data
#     merged_data = world.merge(merged_data, left_on='name', right_on='Country', how='left')
    
#     # Plot the world map
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
#     merged_data.plot(column='Monetary', ax=ax, legend=True, cmap='coolwarm', linewidth=0.8, edgecolor='0.8', legend_kwds={'label': "Monetary", 'orientation': "horizontal"})
    
#     plt.title('World Map of Monetary Values')
#     st.pyplot(plt)
#     plt.clf()
#     plt.close()

def rfmReccomendation():
    st.subheader("Prediction Based On RFM Labels")
    data = user_input("Enter New Inventory Data")
    
    if data is not None:
        data.drop(columns=['product_name','category','about_product','user_id','user_name','review_id','review_title','review_content','img_link','product_link'],inplace=True)
        data.dropna()
        
        st.subheader("New Inventory Data")
        st.dataframe(data)

        data['rating_count'] = data['rating_count'].str.replace(',', '').astype('float')
        data['rating'] = data['rating'].str.replace('|', '0').astype('float')
        data['discount_percentage'] = data['discount_percentage'].str.replace('%', '').astype('float')
        data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype('float')
        data['discounted_price'] = data['discounted_price'].str.replace('₹', '').str.replace(',', '').astype('float')

        ##For Lost Customers
        # Calculate the 80th percentile for each column
        discount_pct_80 = data['discount_percentage'].quantile(0.8)
        rating_pct_80 = data['rating'].quantile(0.8)
        rating_count_pct_80 = data['rating_count'].quantile(0.8)

        # Filter the DataFrame to include only the top 10% of each column
        forLostCustomers = data[
            (data['discount_percentage'] >= discount_pct_80) &
            (data['rating'] >= rating_pct_80) &
            (data['rating_count'] >= rating_count_pct_80)
        ]
        with st.expander("Prediction For Lost Customers"):
            st.dataframe(forLostCustomers)
        
        ##For Big Spenders
        forBigSpenders = data[
        (data['actual_price'] > data['actual_price'].quantile(0.7)) &
        (data['rating'] > data['rating'].quantile(0.7)) &
        (data['rating_count'] > data['rating_count'].quantile(0.7))
        ]

        with st.expander("Prediction For Big Spenders"):
            st.dataframe(forBigSpenders)
        
        ##For Loyal Customers
        
        forLoyalCustomers = data[
        (data['rating'] > data['rating'].quantile(0.1)) &
        (data['rating'] < data['rating'].quantile(0.5)) &
        (data['discount_percentage'] >= (data['discount_percentage'].quantile(0.4)))&
        (data['discount_percentage'] <= (data['discount_percentage'].quantile(0.8)))]  

        with st.expander("Prediction For Loyal Customers"):
            st.dataframe(forLoyalCustomers)
        
        ##For Almost Lost Customers
        forAlmostLost = data[
        (data['discount_percentage'] >= (data['discount_percentage'].quantile(0.9)))
        ]

        with st.expander("Prediction For Almost Customers"):
            st.dataframe(forAlmostLost)

        ##For best Customers
            
        forBestCustomer = data[
        (data['discount_percentage'] >= (data['discount_percentage'].quantile(0.1)))&
        (data['discount_percentage'] <= (data['discount_percentage'].quantile(0.5)))&
        (data['rating_count'] >= (data['rating_count'].quantile(0.1)))&
        (data['rating_count'] <= (data['rating_count'].quantile(0.5)))
        ]

        with st.expander("Prediction For Best Customers"):
            st.dataframe(forBestCustomer)
    else:
        st.success("Enter New Inventory Data")

def run():
    
    labelpath= cn.RfmLabelEncoderPath
    
    data = take_user_input("Enter invoice data")
    
    if data is not None:
        with st.spinner():
            rfm(data,labelpath)
            rfmReccomendation()
    else:
        st.success("Enter Invoice Data")
    