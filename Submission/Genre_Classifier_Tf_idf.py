import json
import re
import string

import nltk
import numpy as np
# Lemmatizer
from nltk.stem import WordNetLemmatizer
# Tokenizer
from nltk.tokenize import TreebankWordTokenizer
from numpy import argmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# Performance metric
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
ENGLISH_STOP_WORDS = stopwords.words('english')


tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

lr = LogisticRegression(C = 1000, max_iter=250, solver='saga') #tf-idf
#lr = LogisticRegressionCV(max_iter=250, solver='saga', cv= 10)
#lr = LogisticRegression(C = 100, max_iter=250, solver='saga')
model = OneVsRestClassifier(lr)

mlb = MultiLabelBinarizer()
onehot_encoder = OneHotEncoder(sparse=False)

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# Encoding the y value
Y_bin = onehot_encoder.fit_transform(np.array(Y).reshape(len(Y), 1))
#Yval_bin = onehot_encoder.fit_transform(np.array(Y_train).reshape(len(Y_train), 1))

#The data is split keeping the class distribution same in both train and val split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_bin,
                                                    train_size=0.9,
                                                    random_state=42,
                                                    stratify=Y)
# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in ENGLISH_STOP_WORDS]
    return ' '.join(no_stopword_text)

def lemmatization(tokens):
    lematized_text = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lematized_text)

def normalization(corpus):
    x = []
    for doc in corpus:
        doc = doc.lower()
        doc = re.sub(r'\d+', '', doc)
        doc = re.sub(r'\n+', ' ', doc)
        doc = remove_stopwords(doc)
        doc = doc.translate(str.maketrans("","", string.punctuation))
        doc = doc.strip()
        tokens = tokenizer.tokenize(doc)
        #doc = lemmatization(tokens)
        x.append(doc)
        #print(x)
    return x

# Tf-df Method
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(X_train)
xval_tfidf = tfidf_vectorizer.transform(X_val)

# fit model on train data
model.fit(xtrain_tfidf, Y_train)

# predict probabilities
y_pred_prob = model.predict_proba(xval_tfidf)

t = 0.5 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

# evaluate performance
print(f1_score(Y_val, y_pred_new, average="macro"))

model.fit(tfidf_vectorizer.transform(X), Y)

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
X_test = test_data['X']

Ytest_prob = model.predict_proba(tfidf_vectorizer.transform(X_test))


Ytest_bin = np.zeros((len(Ytest_prob), 4))
for prob in range(0, len(Ytest_prob)):
    Ytest_bin[prob][argmax(Ytest_prob[prob])] = 1

Y_test_pred = onehot_encoder.inverse_transform(X=Ytest_bin)
# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()