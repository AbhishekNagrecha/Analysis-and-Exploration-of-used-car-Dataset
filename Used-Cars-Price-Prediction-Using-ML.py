#!/usr/bin/env python
# coding: utf-8

# # Used-Cars-Price-Prediction-Using-ML

# In[404]:


#First I import all the essential libraries
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[405]:


#Now input the used car dataset
training_data = pd.read_csv('C:/Users/Abhishek Nagrecha/Desktop/input/train-data.csv')
testing_data = pd.read_csv('C:/Users/Abhishek Nagrecha/Desktop/input/test-data.csv')


# In[406]:


#This is our training data information
training_data.info()


# In[407]:


training_data.head(4)


# In[408]:


training_data.tail(4)


# In[409]:


training_data = training_data.iloc[:,1:]
training_data.head()


# In[410]:


training_data.describe()


# In[411]:


training_data.shape


# In[412]:


training_data['Kilometers_Driven'].value_counts()


# In[413]:


print(training_data['Name'].unique())
print(training_data['Location'].unique())
print(training_data['Fuel_Type'].unique())
print(training_data['Transmission'].unique())
print(training_data['Owner_Type'].unique())


# In[414]:


for i in range(training_data.shape[0]):
    training_data.at[i, 'Company'] = training_data['Name'][i].split()[0]


# In[415]:


#Now here I am checking for the missing values in the dataset
training_data.isnull().sum()


# In[416]:


plt.figure(figsize=(30,20))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)

plt.subplot(141)
plt.title('Location',fontsize=20)
training_data['Location'].value_counts().plot.pie(autopct="%1.1f%%")

plt.subplot(142)
plt.title('Fuel_Type',fontsize=20)
training_data['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%')

plt.subplot(143)
plt.title('Transmission',fontsize=20)
training_data['Transmission'].value_counts().plot.pie(autopct='%1.1f%%')

plt.subplot(144)
plt.title('Owner_Type',fontsize=20)
training_data['Owner_Type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()


# In[417]:


fig = plt.figure(figsize=(20,18))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.add_subplot(2,2,1)
g1 = sns.countplot(x='Location', data=training_data)
loc,labels = plt.xticks()
g1.set_xticklabels(labels,rotation=90)
fig.add_subplot(2,2,2)
g2 = sns.countplot(x='Fuel_Type', data=training_data)
loc,labels = plt.xticks()
g2.set_xticklabels(labels,rotation=0)
fig.add_subplot(2,2,3)
g3 = sns.countplot(x='Seats', data=training_data)
loc,labels = plt.xticks()
g3.set_xticklabels(labels,rotation=0)
fig.add_subplot(2,2,4)
g4 = sns.countplot(x='Owner_Type', data=training_data)
loc,labels = plt.xticks()
g4.set_xticklabels(labels,rotation=0)
plt.show()


# In[418]:


training_data['Price'].describe()


# In[419]:


#Prices decreases progressively as ownership level increases
df_vis_1 = pd.DataFrame(training_data.groupby('Owner_Type')['Price'].mean())
df_vis_1.plot.bar()


# In[420]:


#Prices of used cars on the basis of different fuel types
df_vis_1 = pd.DataFrame(training_data.groupby('Fuel_Type')['Price'].mean())
df_vis_1.plot.bar()


# In[421]:


fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
ax1 = fig.add_subplot(2,2,1)
plt.xlim([0, 100000])
p1 = sns.scatterplot(x="Kilometers_Driven", y="Price", data=training_data)
loc, labels = plt.xticks()
ax1.set_xlabel('Kilometer')

ax2 = fig.add_subplot(2,2,2)
p2 = sns.scatterplot(x="Mileage", y="Price", data=training_data)
loc, labels = plt.xticks()
ax2.set_xlabel('Mileage')

ax3 = fig.add_subplot(2,2,3)
p3 = sns.scatterplot(x="Engine", y="Price", data=training_data)
loc, labels = plt.xticks()
ax3.set_xlabel('Engine')

ax4 = fig.add_subplot(2,2,4)
p4 = sns.scatterplot(x="Power", y="Price", data=training_data)
loc, labels = plt.xticks()
ax4.set_xlabel('Power')

plt.show()


# In[422]:


fig = plt.figure(figsize=(20,6))
ax1 = fig.add_subplot(1,3,1)
sns.boxplot(x='Seats', y='Price', data=training_data)
ax1.set_title('Car Seats vs Car Price')

ax2 = fig.add_subplot(1,3,2)
sns.boxplot(x='Transmission', y='Price', data=training_data)
ax2.set_title('Transmission Type vs Car_Price')

ax3 = fig.add_subplot(1,3,3)
sns.boxplot(x='Fuel_Type', y='Price', data=training_data)
ax3.set_title('Car Fuel_type vs Car_Price')

plt.show()


# In[423]:


training_data.at[i, 'Company'] = training_data['Name'][i].split()[0]
training_data.at[i, 'Mileage(km/kg)'] = training_data['Mileage'][i].split()[0]
training_data.at[i, 'Engine(CC)'] = training_data['Engine'][i].split()[0]
training_data.at[i, 'Power(bhp)'] = training_data['Power'][i].split()[0]


# In[424]:


training_data['Mileage(km/kg)'] = training_data['Mileage(km/kg)'].astype(float)
training_data['Engine(CC)'] = training_data['Engine(CC)'].astype(float)


# In[425]:


training_data['Power'][76]


# In[426]:


x = 'n'
count = 0
position = []
for i in range(training_data.shape[0]):
    if training_data['Power(bhp)'][i]=='null':
        x = 'Y'
        count = count + 1
        position.append(i)
print(x)
print(count)
print(position)


# In[427]:


training_data = training_data.drop(training_data.index[position])
training_data = training_data.reset_index(drop=True)
training_data.shape


# In[428]:


training_data['Power(bhp)'] = training_data['Power(bhp)'].astype(float)
training_data.head()


# In[429]:


for i in range(training_data.shape[0]):
    if pd.isnull(training_data.loc[i,'New_Price']) == False:
        training_data.at[i,'New_car_Price'] = training_data['New_Price'][i].split()[0]
        


# In[430]:


training_data['New_car_Price'] = training_data['New_car_Price'].astype(float)


# In[431]:


training_data.drop(["Name"],axis=1,inplace=True)
training_data.drop(["Mileage"],axis=1,inplace=True)
training_data.drop(["Engine"],axis=1,inplace=True)
training_data.drop(["Power"],axis=1,inplace=True)
training_data.drop(["New_Price"],axis=1,inplace=True)


# # LABEL AND ONE HOT ENCODING

# In[ ]:





# In[432]:


var = 'Location'
training_data[var].value_counts()


# In[433]:


#Now one-hot-encoding the variable location
Location = training_data[[var]]
Location = pd.get_dummies(Location,drop_first=True)
Location.head()


# In[434]:


var = 'Fuel_Type'
training_data[var].value_counts()


# In[435]:


#Now one-hot-encoding the variable Fuel_type
Fuel_type = training_data[[var]]
Fuel_type = pd.get_dummies(Fuel_type,drop_first=True)
Fuel_type.head()


# In[436]:


var = 'Transmission'
training_data[var].value_counts()


# In[437]:


#Now one-hot-encoding the variable Fuel_type

Transmission = training_data[[var]]
Transmission = pd.get_dummies(Transmission,drop_first=True)
Transmission.head()


# In[438]:


var = 'Owner_Type'
training_data[var].value_counts()


# In[439]:


#Label encoded this as it had ordered data
training_data.replace({"First":1,"Second":2,"Third": 3,"Fourth & Above":4},inplace=True)
training_data.head()


# In[440]:


var = 'Company'
training_data[var].value_counts()


# In[441]:


final_train= pd.concat([training_data,Location,Fuel_type,Transmission],axis=1)
final_train.head()


# In[442]:


training_data.drop(["Company"],axis=1,inplace=True)


# In[443]:


final_train= pd.concat([training_data,Location,Fuel_t,Transmission],axis=1)
final_train.head()


# In[444]:


final_train.shape


# In[445]:


#Now here I have droppped ess important features from the dataset
final_train.drop(["Location","Fuel_Type","Transmission","New_car_Price"],axis=1,inplace=True)
final_train.head() 


# In[446]:


final_train.shape


# # Now working on our testing_data to apply the ML algorithms

# In[447]:


testing_data.head()


# In[448]:


testing_data = testing_data.iloc[:,1:]

print("Shape of test data Before dropping any Row: ",testing_data.shape)
testing_data = testing_data[testing_data['Mileage'].notna()]
print("Shape of test data After dropping Rows with NULL values in Mileage: ",testing_data.shape)
testing_data = testing_data[testing_data['Engine'].notna()]
print("Shape of test data After dropping Rows with NULL values in Engine : ",testing_data.shape)
testing_data = testing_data[testing_data['Power'].notna()]
print("Shape of test data After dropping Rows with NULL values in Power  : ",testing_data.shape)
testing_data = testing_data[testing_data['Seats'].notna()]
print("Shape of test data After dropping Rows with NULL values in Seats  : ",testing_data.shape)
print('Droping null done')

testing_data = testing_data.reset_index(drop=True)
print('Index reset done')

for i in range(testing_data.shape[0]):
    testing_data.at[i, 'Mileage(km/kg)'] = testing_data['Mileage'][i].split()[0]
    testing_data.at[i, 'Engine(CC)'] = testing_data['Engine'][i].split()[0]
    testing_data.at[i, 'Power(bhp)'] = testing_data['Power'][i].split()[0]
print('Split Done') 

testing_data['Mileage(km/kg)'] = testing_data['Mileage(km/kg)'].astype(float)
testing_data['Engine(CC)'] = testing_data['Engine(CC)'].astype(float)
print('casting 1 Done') 

position = []
for i in range(testing_data.shape[0]):
    if testing_data['Power(bhp)'][i]=='null':
        position.append(i)
        
testing_data = testing_data.drop(testing_data.index[position])
testing_data = testing_data.reset_index(drop=True) 

testing_data['Power(bhp)'] = testing_data['Power(bhp)'].astype(float)
print('casting 2 Done') 


for i in range(testing_data.shape[0]):
    if pd.isnull(testing_data.loc[i,'New_Price']) == False:
        testing_data.at[i,'New_car_Price'] = testing_data['New_Price'][i].split()[0]

testing_data['New_car_Price'] = testing_data['New_car_Price'].astype(float)
testing_data.drop(["Name"],axis=1,inplace=True)
testing_data.drop(["Mileage"],axis=1,inplace=True)
testing_data.drop(["Engine"],axis=1,inplace=True)
testing_data.drop(["Power"],axis=1,inplace=True)
testing_data.drop(["New_Price"],axis=1,inplace=True)

var = 'Location'
Location = testing_data[[var]]
Location = pd.get_dummies(Location,drop_first=True)
Location.head()

var = 'Fuel_Type'
Fuel_t = testing_data[[var]]
Fuel_t = pd.get_dummies(Fuel_t,drop_first=True)
Fuel_t.head()

var = 'Transmission'
Transmission = testing_data[[var]]
Transmission = pd.get_dummies(Transmission,drop_first=True)
Transmission.head()

testing_data.replace({"First":1,"Second":2,"Third": 3,"Fourth & Above":4},inplace=True)
testing_data.head()

final_test= pd.concat([testing_data,Location,Fuel_t,Transmission],axis=1)
final_test.head()

final_test.drop(["Location","Fuel_Type","Transmission","New_car_Price"],axis=1,inplace=True)
final_test.head()

print("Final Test Size: ",final_test.shape)


# # Now selection the final features from the dataset

# In[449]:


final_train.columns


# In[450]:


X = final_train.loc[:,['Year', 'Kilometers_Driven', 'Owner_Type', 'Seats',
       'Mileage(km/kg)', 'Engine(CC)', 'Power(bhp)', 
       'Location_Bangalore', 'Location_Chennai', 'Location_Coimbatore',
       'Location_Delhi', 'Location_Hyderabad', 'Location_Jaipur',
       'Location_Kochi', 'Location_Kolkata', 'Location_Mumbai',
       'Location_Pune', 'Fuel_Type_Diesel', 'Fuel_Type_LPG',
       'Fuel_Type_Petrol', 'Transmission_Manual']]
X.shape


# In[451]:


y = final_train.loc[:,['Price']]
y.head(10)


# In[458]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)


# In[459]:


plt.figure(figsize=(18,18))
sns.heatmap(final_train.corr(),annot=True,cmap='RdYlGn')

plt.show()


# In[469]:


from sklearn.ensemble import ExtraTreesRegressor
selection= ExtraTreesRegressor(n_estimators=10)
selection.fit(X,y)


# In[470]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred= linear_reg.predict(X_test)
print("Accuracy on Traing set: ",linear_reg.score(X_train,y_train))
print("Accuracy on Testing set: ",linear_reg.score(X_test,y_test))


# In[ ]:




