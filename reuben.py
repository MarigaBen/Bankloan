#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
train = pd.read_csv("trainingdata.csv") 
train.head()


# In[17]:


train['Married'].replace({'Yes':1,'No':0},inplace=True)
train


# In[90]:


train['Education'].replace({'Graduate':1,'Not Graduate':0},inplace=True)
train


# In[92]:


train['Married'].replace({'Yes':1,'No':0},inplace=True)
train.head()


# In[95]:


train['Married'].replace({'Yes':1,'No':0},inplace=True)
train['Property_Area'].replace({'Urban':3,'Semiurban':2,'Rural':1},inplace=True)
train[['Education','Married','ApplicantIncome']]
train


# In[94]:


train['Loan_Status'].replace({'Y':1,'N':0},inplace=True)
train


# In[75]:


#We know that machine learning models take only numbers as inputs and can not process strings. 
#So, we have to deal with the categorical features present in the dataset and convert them into numbers.
train.isnull().sum()


# In[96]:


train = train.dropna()
train.isnull().sum()


# In[97]:


X = train[['Married','ApplicantIncome','LoanAmount','Credit_History']]
y = train.Loan_Status
X.shape, y.shape


# In[98]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)


# In[99]:


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state= 10) 
model.fit(x_train, y_train)


# In[100]:


#Now, our model is trained, let’s check its performance on both the training and validation set:

from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)


# In[101]:


#The model is 80% accurate on the validation set. Let’s check the performance on the training set too:

pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)


# In[102]:


# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# In[103]:


#We have to create the python script for our app. 
#Let me show the code first and then I will explain it to you in detail:
import pickle
import streamlit as st
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
@st.cache()
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Married, ApplicantIncome,LoanAmount, Credit_History):   
# Pre-processing user input    
 
    if Married == "Unmarried":
        Married = 0
    else:
        Married = 1

    if Credit_History == "Unclear Debts":
        Credit_History = 0
    else:
        Credit_History = 1  

    LoanAmount = LoanAmount / 1000

# Making predictions 
    prediction = classifier.predict([[Married, ApplicantIncome, LoanAmount, Credit_History]])

    if prediction == 0:
        pred = 'Rejected'
    else:
         pred = 'Approved'
    return pred


# In[104]:


#this is the main function in which we define our webpage  
def main():#front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
    #display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    #following lines create boxes in which user can enter data required to make prediction 
    Married = st.selectbox('Marital Status',("Unmarried","Married")) 
    ApplicantIncome = st.number_input("Applicants monthly income") 
    LoanAmount = st.number_input("Total loan amount")
    Credit_History = st.selectbox('Credit_History',("Unclear Debts","No Unclear Debts"))
    result =""
    #when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
       result = prediction(Married, ApplicantIncome, 
       LoanAmount, Credit_History) 
       st.success('Your loan is {}'.format(result))
       print(LoanAmount)

    if __name__=='__main__': 
        main()

