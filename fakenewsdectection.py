!pip install nltk
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

news_df=pd.read_csv('train.csv')
news_df.head()

news_df.shape

news_df.isna().sum()   

news_df=news_df.fillna(' ') 
news_df.isna().sum()

news_df['content']=news_df['author']+news_df['title']
news_df

news_df['content'][20796]

# stemming
ps=PorterStemmer()



def stemming(content):
    if content is None:
        content = "" 
    elif not isinstance(content, str):
        content = str(content)  

    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)

news_df['content']

X=news_df['content'].values
Y=news_df['label'].values
print(X)

vector=TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)
print(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
X_train.shape

model=LogisticRegression()
model.fit(X_train,Y_train)

with open('fakenewsdetection.pkl','wb') as file: 
    pickle.dump('model',file)

train_y_pred=model.predict(X_train)
train_accuracy=accuracy_score(train_y_pred,Y_train)
print(train_accuracy)

test_y_pred=model.predict(X_test)
test_accuracy=accuracy_score(test_y_pred,Y_test)
print(test_accuracy)

# prediction
x_new=X_test[2]
prediction=model.predict(x_new)
if prediction[0]==0:
  print('The news is real')
else:
  print('The news is fake')
