import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

class my_model():
    def __init__(self, loss="perceptron", learning_rate_type="adaptive", initial_learning_rate=0.1, shuffle=False):
        self.loss = loss
        self.learning_rate_type = learning_rate_type
        self.initial_learning_rate = initial_learning_rate
        self.shuffle = shuffle

        self.model = SGDClassifier(loss=self.loss, verbose=1, learning_rate=self.learning_rate_type, eta0=self.initial_learning_rate, shuffle=self.shuffle)

        self.data_preprocessor = CountVectorizer(stop_words='english', binary=True)

    def fit(self, X, y):
        x = self.data_preprocessor.fit_transform(X["description"])
        x = pd.DataFrame(x.toarray())
        self.model.fit(x, y)
        return

    def predict(self, X):
        x = self.data_preprocessor.transform(X["description"])
        x = pd.DataFrame(x.toarray())
        predictions = self.model.predict(x)
        return predictions