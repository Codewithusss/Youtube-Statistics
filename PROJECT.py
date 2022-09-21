# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:14:12 2022

@author: Codewithusss
"""
#Importing Libraries
import pandas as pd
import numpy as np

#Reading file
data = pd.read_csv(r"E:\Clg\videosstats.csv")
data


data.head()
print()
selected_features = ['Title','Video ID','Published At','Keyword','Likes','Comments','Views']
print(selected_features)

for feature in selected_features:
    data[feature] = data[feature].fillna('')

#Quries
user = int(input("\nEnter no. of Likes\n"))
print("Data for "+str(user)+" no. of Likes")
print(data[data.Likes==user])
print("\nData of Technical Videoes\n")
print(data[data.Keyword=='tech'])
user2 = int(input("\nEnter no. of Views\n"))
print("\nData for "+str(user2)+" no. of Views")
print(data[data.Views==user2])
print("\n")


#SENTIMENTAL ANALYSIS , POPULARITY ANALYSIS

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv("E:\ClgInternship\comments.csv")
dataset

nltk.download('stopwords')

corpus = []
for i in range(0,1800):
    review = re.sub('[^a-zA-Z]',' ', dataset['Comment'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
print("\nDisplaying Corpus data\n") 
corpus

#Data Transformation
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 18000)
X = cv.fit_transform(corpus).toarray()
#y = dataset.iloc[:,-1].values

y11 = []
for i in range(0,1800):
    rec0=dataset.iloc[:,-1]
    y11.append(rec0[i])
 
    
#Traning and Testing 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

print("\nShape of dataset")    
dataset.shape


#Dividing dataset into Training and Testset
X_train, X_test, y_train, y_test = train_test_split(X, y11, test_size = 0.3, random_state = 0)
 #in   
classifier =  GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#MODEL PERFORMANCE
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("\n\n")
print(cm)
print("\nAccuracy Score")
accuracy_score(y_test, y_pred)

#Queries
data
n=data.shape
n=n[0]
print("\nDescribing Views")
data['Views'].describe()

status=[]
for i in range(n):
    rec=data.iloc[i]
    if(rec[7]>10000000):
        r='#Trending10!!'
    elif(rec[7]>5000000):
        r='#Trending15!!'
    else:
        r='Average'
    
    status.append(r)

data["Status"] = status
print("\nAppending Status Column\n")
print(data)



#LINEAR REGRESSION ALGORITHM

import matplotlib.pyplot as plt

dataset2 = pd.read_csv("E:\ClgInternship\likevsviews.csv")
print("\nDATASET2\n")
print(dataset2)

#Assigning data
print("\nValues of X")
X = dataset2.iloc[:,:-1].values
#rows  excepct last c
print(X)
print("\nValues of y")
y = dataset2.iloc[:,-1].values
#with last column
print(y)

#Spliting data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 12)
print("\n X_train size \t X_test size")
print(X_train.shape,"\t\t",X_test.shape)
print("\n y_train size \t y_test size")
print(y_train.shape,"\t\t\t",y_test.shape)


# Training the Simple Linear Regression model on the Training set by passing x values and y values
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#getting coeff and intercept values by algorithm for given x_train and y_train data
m=regressor.coef_
print("\nCoefficient = ",m)
c=regressor.intercept_
print("\nIntercept = ",c)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Visualising the Training set results
#Graph
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Views vs Likes')
plt.xlabel('No. of Views')
plt.ylabel('No. of Likes')
plt.show()


##Graph
from matplotlib import pyplot as plt

xset = dataset2['Likes'].astype('string')
yset = dataset2['Views'].astype('string')

plt.title("BAR")
plt.xlabel("Likes")
plt.ylabel("Views")
plt.bar(xset,yset)
plt.show()

plt.xlabel("Likes")
plt.ylabel("Views")
plt.barh(xset,yset,color=["pink","green","black","yellow","red","blue"])
plt.show()

#QUERY
topCat_df = pd.pivot_table(data, values = "Views", index="Keyword", aggfunc= np.sum)[:5]
print ("\nTop 5 Trending\n")
print(topCat_df)