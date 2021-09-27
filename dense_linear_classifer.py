import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import TreebankWordTokenizer
import math
from sklearn.model_selection import GridSearchCV

# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}

#initialize a tokenzier, missing_value
tokenizer = TreebankWordTokenizer()
missing_value = 0.0

# convert a document into a vector
def document_to_vector(corpus):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    vec = []
    # Tokenize the input documents
    for doc in corpus:
        tokens = tokenizer.tokenize(doc)
        vectors = []
        for token in tokens:
            if token in w2v:
                vectors.append(w2v.get(token))
        
        # Aggregate the vectors of words in the input document
        vec.append(np.sum(np.array(vectors), axis = 0) / len(vectors))

    return vec
            

# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    #TODO: convert each of the training documents into a vector
    Xtr_vec = document_to_vector(Xtr)
    #TODO: train the logistic regression classifier
    log_reg = LogisticRegression(C=C)
    model = log_reg.fit(Xtr_vec, Ytr)
    return model

# fit a linear model 
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    #TODO: convert each of the testing documents into a vector
    Xtst_vec = document_to_vector(Xtst)
    
    #TODO: test the logistic regression classifier and calculate the accuracy
    score = model.score(Xtst_vec, Ytst)
    return score


# TODO: search for the best C parameter using the validation set
#K = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
#for k in K:
#    C = math.pow(3, k)
#    model = fit_model(X_train, Y_train, C)
#    score = test_model(model, X_val, Y_val)
#   print("K: " + str(k) + " Score: " + str(score))

model = LogisticRegression()
solvers = ['sag', 'saga']#'newton-cg', 
penalty = ['l2']
K = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
iterations = [40, 50, 60, 75, 100]
c_values = [math.pow(3, k) for k in K]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values, max_iter = iterations)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(document_to_vector(X_train), Y_train)

# summarize results
print("Training Fraction: %f Best: %f using %s" % (train_frac, grid_result.best_score_, grid_result.best_params_))
print("Val Accuracy: %f" % grid_search.score(document_to_vector(X_val), Y_val))

# TODO: fit the model to the concatenated training and validation set
#   test on the test set and print the result

# compute the train, val, test splits
tr_frac = 0.8
tr_end = int(tr_frac*len(Xr))


# store the train test splits
X_tr = Xr[0:tr_end]
Y_tr = Yr[0:tr_end]
X_ts = Xr[tr_end:]
Y_ts = Yr[tr_end:]
grid_result = grid_search.fit(document_to_vector(X_tr), Y_tr)
# summarize results
print("Training Fraction: %f Best: %f using %s" % (tr_frac, grid_result.best_score_, grid_result.best_params_))
print("Final Test Accuracy: %f" % grid_search.score(document_to_vector(X_ts), Y_ts))


