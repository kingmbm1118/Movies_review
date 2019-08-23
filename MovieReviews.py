# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:06:29 2018

@author: Mahmoud
"""

# Assignment: movie reviews competition on Kaggle
# Develop and implement a method for the Kaggle challenge 
# (https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) 
# and submit your results on Kaggle 
# Prepare a presentation on the method you use, your results, your observations, ….
# Email the presentation to melsaban@raisaenergy.com and 
# be ready to present it in a face-2-face interview 



# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.tsv', delimiter = '\t', quoting = 3)
    
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 156060):
    Phrase = re.sub('[^a-zA-Z]', ' ', dataset['Phrase'][i])
    Phrase = Phrase.lower()
    Phrase = Phrase.split()
    ps = PorterStemmer()
    Phrase = [ps.stem(word) for word in Phrase if not word in set(stopwords.words('english'))]
    Phrase = ' '.join(Phrase)
    corpus.append(Phrase)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
#y = dataset.iloc[:, 1].values
y= dataset['Sentiment'].values

 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
mse = ((y_pred - y_test) ** 2).mean()

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Predicting the Test set results for the test dataset provided from kaggle
validate_data = pd.read_csv('test.tsv', delimiter = '\t', quoting = 3)
validate_data = validate_data[['PhraseId','Phrase']]


validate_corpus = []
for i in range(0, 66292):
    validate_Phrase = re.sub('[^a-zA-Z]', ' ', validate_data['Phrase'][i])
    validate_Phrase = validate_Phrase.lower()
    validate_Phrase = validate_Phrase.split()
    validate_Phrase = [ps.stem(word) for word in validate_Phrase if not word in set(stopwords.words('english'))]
    validate_Phrase = ' '.join(validate_Phrase)
    validate_corpus.append(validate_Phrase)

Validate_X = cv.fit_transform(validate_corpus).toarray()
Validate_y_pred = classifier.predict(Validate_X)

validate_data['Sentiment'] = Validate_y_pred
validate_data.to_csv('validate_data.tsv', sep='\t', encoding='utf-8', index=False)


# Submission Format
validate_data = validate_data[['PhraseId','Sentiment']]
validate_data.to_csv('validate_data.csv', sep=',', encoding='utf-8', index=False)

