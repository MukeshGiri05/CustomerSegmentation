# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, āhere's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

import time, warnings
import datetime as dt

#visualizations
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

warnings.filterwarnings("ignore")

# %% [markdown]
# # Get the Data

# %%


# %%
#load the dataset
retail_df = pd.read_csv('data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
retail_df.head()

# %% [markdown]
# # Prepare the Data

# %% [markdown]
# 
# As customer clusters may vary by geography, I’ll restrict the data to only United Kingdom customers, which contains most of our customers historical data.

# %%
retail_uk = retail_df[retail_df['Country']=='United Kingdom']
#check the shape
retail_uk.shape

# %%
retail_df

# %%
#remove canceled orders
retail_uk = retail_uk[retail_uk['Quantity']>0]
retail_uk.shape

# %%
#remove rows where customerID are NA
retail_uk.dropna(subset=['CustomerID'],how='all',inplace=True)
retail_uk.shape

# %%
#restrict the data to one full year because it's better to use a metric per Months or Years in RFM
retail_uk = retail_uk[retail_uk['InvoiceDate']>= "2010-12-09"]
retail_uk.shape

# %%
print("Summary..")
#exploring the unique values of each attribute
print("Number of transactions: ", retail_uk['InvoiceNo'].nunique())
print("Number of products bought: ",retail_uk['StockCode'].nunique())
print("Number of customers:", retail_uk['CustomerID'].nunique() )
print("Percentage of customers NA: ", round(retail_uk['CustomerID'].isnull().sum() * 100 / len(retail_df),2),"%" )

# %% [markdown]
# # RFM Analysis

# %% [markdown]
# **RFM** (Recency, Frequency, Monetary) analysis is a customer segmentation technique that uses past purchase behavior to divide customers into groups. 
# RFM helps divide customers into various categories or clusters to identify customers who are more likely to respond to promotions and also for future personalization services.
# 
# - RECENCY (R): Days since last purchase
# - FREQUENCY (F): Total number of purchases
# - MONETARY VALUE (M): Total money this customer spent.
# 
# We will create those 3 customer attributes for each customer.

# %% [markdown]
# ## Recency

# %% [markdown]
# To calculate recency, we need to choose a date point from which we evaluate **how many days ago was the customer's last purchase**.

# %%
#last date available in our dataset
retail_uk['InvoiceDate'].max()

# %% [markdown]
# The last date we have is 2011-12-09 so we will use it as reference.

# %%
now = dt.date(2011,12,9)
print(now)

# %%
#create a new column called date which contains the date of invoice only
retail_uk['date'] = pd.DatetimeIndex(retail_uk['InvoiceDate']).date

# %%
retail_uk.head()

# %%
#group by customers and check last date of purshace
recency_df = retail_uk.groupby(by='CustomerID', as_index=False)['date'].max()
recency_df.columns = ['CustomerID','LastPurshaceDate']
recency_df.head()

# %%

#calculate recency
recency_df['Recency'] = recency_df['LastPurshaceDate'].apply(lambda x: (now - x).days)

# %%
recency_df.head()

# %%
#drop LastPurchaseDate as we don't need it anymore
recency_df.drop('LastPurshaceDate',axis=1,inplace=True)

# %% [markdown]
# ## Frequency

# %% [markdown]
# Frequency helps us to know how many times a customer purchased from us. To do that we need to check **how many invoices are registered by the same customer**.

# %%
# drop duplicates
retail_uk_copy = retail_uk
retail_uk_copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
#calculate frequency of purchases
frequency_df = retail_uk_copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
frequency_df.columns = ['CustomerID','Frequency']
frequency_df.head()

# %% [markdown]
# ## Monetary

# %% [markdown]
# Monetary attribute answers the question: **How much money did the customer spent over time?**
# 
# To do that, first, we will create a new column total cost to have the total price per invoice.

# %%
#create column total cost
retail_uk['TotalCost'] = retail_uk['Quantity'] * retail_uk['UnitPrice']

# %%
monetary_df = retail_uk.groupby(by='CustomerID',as_index=False).agg({'TotalCost': 'sum'})
monetary_df.columns = ['CustomerID','Monetary']
monetary_df.head()

# %% [markdown]
# ## Create RFM Table

# %%
#merge recency dataframe with frequency dataframe
temp_df = recency_df.merge(frequency_df,on='CustomerID')
temp_df.head()

# %%
#merge with monetary dataframe to get a table with the 3 columns
rfm_df = temp_df.merge(monetary_df,on='CustomerID')
#use CustomerID as index
rfm_df.set_index('CustomerID',inplace=True)
#check the head
rfm_df.head()
temp = rfm_df.copy()

# %% [markdown]
# ## RFM Table Correctness verification

# %%
retail_uk[retail_uk['CustomerID']=='12820']

# %%
(now - dt.date(2011,9,26)).days == 74

# %% [markdown]
# ## Customer segments with RFM Model

# %% [markdown]
# The simplest way to create customers segments from RFM Model is to use** Quartiles**. We assign a score from 1 to 4 to Recency, Frequency and Monetary. Four is the best/highest value, and one is the lowest/worst value. A final RFM score is calculated simply by combining individual RFM score numbers.
# 
# Note: Quintiles (score from 1-5) offer better granularity, in case the business needs that but it will be more challenging to create segments since we will have 555 possible combinations. So, we will use quartiles.

# %% [markdown]
# ### RFM Quartiles

# %%
quantiles = rfm_df.quantile(q=[0.25,0.5,0.75])
quantiles

# %%
quantiles.to_dict()

# %% [markdown]
# ### Creation of RFM Segments

# %% [markdown]
# We will create two segmentation classes since, high recency is bad, while high frequency and monetary value is good.

# %%
# Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

# %%
#create rfm segmentation table
rfm_segmentation = rfm_df
rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,))

# %%
rfm_segmentation.head()

# %% [markdown]
# Now that we have the score of each customer, we can represent our customer segmentation.
# First, we need to combine the scores (R_Quartile, F_Quartile,M_Quartile) together.

# %%
rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) \
                            + rfm_segmentation.F_Quartile.map(str) \
                            + rfm_segmentation.M_Quartile.map(str)
rfm_segmentation.head()

# %% [markdown]
# Best Recency score = 4: most recently purchase. Best Frequency score = 4: most quantity purchase. 
# Best Monetary score = 4: spent the most.

# %% [markdown]
# Let's see who are our** Champions** (best customers).

# %%
rfm_segmentation[rfm_segmentation['RFMScore']=='444'].sort_values('Monetary', ascending=False).head(10)

# %% [markdown]
# We can find [here](http://www.blastam.com/blog/rfm-analysis-boosts-sales) a suggestion of key segments and then we can decide which segment to consider for further study.
# 
# Note: the suggested link use the opposite valuation: 1 as highest/best score and 4 is the lowest.
# 
# **How many customers do we have in each segment?**

# %%
# print("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
# print('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
# print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
# print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
# print('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
# print('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))


data = {'Metrics': ['Best Customers', 'Loyal Customers', 'Big Spenders', 'Almost Lost', 'Lost Customers', 'Lost Cheap Customers'],
        'Count': [len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']),
                  len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]),
                  len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]),
                  len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']),
                  len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']),
                  len(rfm_segmentation[rfm_segmentation['RFMScore']=='111'])]}

df = pd.DataFrame(data)
df.set_index('Metrics',inplace=True)
print(df)
df.to_csv('segments.csv')

# %% [markdown]
# Now that we knew our customers segments we can choose how to target or deal with each segment.
# 
# For example:
# 
# **Best Customers - Champions**: Reward them. They can be early adopters to new products. Suggest them "Refer a friend".
# 
# **At Risk**: Send them personalized emails to encourage them to shop.
# 
# More ideas about what actions to perform in [Ometria](http://54.73.114.30/customer-segmentation#).

# %%
import numpy as np

conditions = [
 (rfm_segmentation['R_Quartile'] >= 4) & (rfm_segmentation['F_Quartile'] >= 4) & (rfm_segmentation['M_Quartile'] >= 4), # 'Best Customers'
 (rfm_segmentation['R_Quartile'] < 4) & (rfm_segmentation['F_Quartile'] >= 4) & (rfm_segmentation['M_Quartile'] >= 4), # 'Loyal Customers'
 (rfm_segmentation['R_Quartile'] < 4) & (rfm_segmentation['F_Quartile'] < 4) & (rfm_segmentation['M_Quartile'] >= 4), # 'Big Spenders'
 (rfm_segmentation['R_Quartile'] > 3) & (rfm_segmentation['F_Quartile'] < 4) & (rfm_segmentation['M_Quartile'] < 4), # 'Almost Lost'
 (rfm_segmentation['R_Quartile'] < 4) & (rfm_segmentation['F_Quartile'] < 4) & (rfm_segmentation['M_Quartile'] < 4), # 'Lost Customers'
]

labels = ['Best Customers', 'Loyal Customers', 'Big Spenders', 'Almost Lost', 'Lost Customers']

# Assign the labels to the RFM values
rfm_segmentation['CustomerSegment'] = np.select(conditions, labels, default='Other')


# %%
rfm_segmentation.to_csv('rfm_segmentation.csv')

# %%
rfm_segmentation['CustomerSegment'].unique()

# %%
temp['CustomerSegment'] = rfm_segmentation['CustomerSegment']

# %%
temp

# %%
temp.to_csv('finalData.csv')

# %%
temp['CustomerSegment'].unique()

# %%


# %% [markdown]
# ## Creating AN ANN
# 

# %% [markdown]
# 

# %%

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
temp['CustomerSegment'] = le.fit_transform(temp['CustomerSegment'])


# %%


# %%
from sklearn.model_selection import train_test_split

X = temp.drop('CustomerSegment', axis=1) # Features
y = temp['CustomerSegment'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from keras.utils import to_categorical

# One-hot encode the target variable
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)



# %%
print(type(X_test))

# %%
np.unique(y_train)

# %%
from keras.models import Sequential
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=10)

loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))


# %%
# Generate predictions for the test data
predictions = model.predict(X_test)
predictions

# %%

# Get the index of the maximum probability
max_prob_indices = np.argmax(predictions, axis=1)

# Transform the indices back to original form
original_labels = le.inverse_transform(max_prob_indices)

# %%
original_labels

# %%
print(len(original_labels))
print(len(X_test))


# %%
X_test['Labels'] =original_labels

# %%
X_test

# %%
X_test.to_csv('testing.csv')

# %%


# %%
model.save('RFMSegmentationModel.keras')
# from keras.models import save_model
# save_model(model, 'RFMSegmentationModel.keras')

# %%
from tensorflow.keras.models import load_model
loadedModel = load_model('RFMSegmentationModel.keras')

# %%
import joblib
# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')


