import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

### our code

import models

# global label encoder and vectorizer
label_encoder = LabelEncoder()
vectorizer = TfidfVectorizer()
class MajorityBaselineClassifier:
    """Baseline model for text classification that classifies every input as the most common class
    in the dataset."""

    def __init__(self) -> None:
        self.name = "Majority baseline classifier"
        self.prediction = "inform"

    def train(self, df_train):
        """Find most common class and set it as the value to be predicted.
        """
        majority_class_label = df_train["dialog_act"].value_counts().idxmax()
        self.prediction = majority_class_label

    def predict_act(self, utterance):
        """Return the dialog act found to be most common during training.
        """
        return self.prediction

    def predict(self, X_test: pd.DataFrame):
        """Return the prediction for each element in a test set.
        """
        return [self.predict_act(element) for element in X_test]

class RuleBaselineClassifier:
    """Baseline model that classifies utterances based on keywords."""

    def __init__(self) -> None:
        self.name = "Rule-based baseline classifier"

    def utterance_contains_word(self, utterance, words):
        """Helper function to determine whether an utterance contains any word in a given list"""
        for word in words:
            if word in utterance:
                return True
        return False
    def predict_act(self, utterance) -> str:
        """Predict the dialog act of utterances using a set of rules."""
        if self.utterance_contains_word(utterance, ["food", "restaurant", "town", "east", "west", "south", "north", "part"]):
            return "inform"
        if ("it" in utterance and "is" in utterance or "does" in utterance and "is" in utterance):
            return "confirm"
        if self.utterance_contains_word(utterance, ["yes", "right"]):
            return "affirm"
        if self.utterance_contains_word(utterance, ["number", "phone", "address", "post"]):
            return "request"
        if "thank" in utterance and "you" in utterance:
            return "thankyou"
        if self.utterance_contains_word(utterance, ["noise", "sil", "unintelligible"]):
            return "null"
        if self.utterance_contains_word(utterance, ["good", "bye", "goodbye"]):
            return "bye"
        if ("how" in utterance and "about" in utterance) or "else" in utterance:
            return "reqalts"
        if "no" in utterance:
            return "negate"
        if self.utterance_contains_word(utterance, ["hi", "hello"]):
            return "hello"
        if (("repeat" in utterance and "that" in utterance) or "repeat" in utterance or "back" in utterance):
            return "repeat"
        if self.utterance_contains_word(utterance, ["okay", "kay"]):
            return "ack"
        if (("start" in utterance and "over" in utterance) or "start" in utterance or "reset" in utterance):
            return "restart"
        if self.utterance_contains_word(utterance, ["wrong", "dont"]):
            return "deny"
        if "more" in utterance:
            return "reqmore"
        else:
            return "inform"  # Can replace this with 'error' later on if necessary for evaluation.

    def predict(self, X_test: pd.DataFrame):
        """Predict every utterance in a given dataframe of features."""
        return [self.predict_act(element) for element in X_test]

class DialogActsClassifier:
    """Machine learning classifier for dialog act classification."""

    def __init__(self, model=None, name="Decision tree classifier"):
        if model is None:
            self.model = DecisionTreeClassifier()
        else:
            self.model = model
        self.name = name

    def train(self, X_train, y_train, hyperparams_dict=None):
        """Fit the label encoder, the bag-of-words vectorizer and train the model."""
        # fit label encoder and transform dependent variable
        label_encoder.fit(y_train)
        y_train_encoded = label_encoder.transform(y_train)

        # fit bag-of-words vectorizer and transform features
        vectorizer.fit(X_train)
        X_train_bow = vectorizer.transform(X_train)

        # if a hyperparameter grid is specified, perform a grid search
        if hyperparams_dict is None:
            self.model.fit(X_train_bow, y_train_encoded)
        else:
            print(f"Performing grid search for {self.name}")
            cv = StratifiedKFold(n_splits=3)
            gridsearch = GridSearchCV(
                estimator=self.model,
                cv=cv,
                scoring="f1_macro",
                param_grid=hyperparams_dict,
                n_jobs=-1,
            )
            gridsearch.fit(X_train_bow, y_train_encoded)
            self.model = gridsearch.best_estimator_

    def predict(self, X_test):
        """Predict dialog acts for test data."""
        X_test_bow = [self.transform_input(utterance) for utterance in X_test]
        X_test_bow = np.array(X_test_bow).reshape(len(X_test_bow), -1)
        return self.model.predict(X_test_bow)

    def predict_act(self, utterance):
        """Predict the dialog act of an utterance."""
        # transform utterance and predict its label
        utterance_bow = vectorizer.transform([utterance])
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

        return predicted_label[0] if len(predicted_label) > 0 else None

    def transform_input(self, utterance: str):
        """Transform the input utterance into a TF-IDF vector with OOV handling."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = vectorizer.transform([utterance])

        # Calculate the average TF-IDF vector for the words in the utterance
        if utterance_bow.nnz > 0:
            average_vector = utterance_bow.sum(axis=0) / utterance_bow.nnz
        else:
            # Handle the case where there are no words in the utterance (empty string)
            average_vector = np.zeros((1, len(vectorizer.get_feature_names_out())))

        return average_vector

class LogisticRegressionClassifier:
    """Logistic Regression classifier for dialog act classification."""

    def __init__(self):
        self.name = "Logistic regression classifier"
        self.model = LogisticRegression()
        self.oov_token = 0  # Special integer for out-of-vocabulary words

    def train(self, X_train, y_train):
        """Train the logistic regression model and the label encoder."""
        label_encoder.fit(y_train)
        y_train_encoded = label_encoder.transform(y_train)
        vectorizer.fit(X_train)
        X_train_bow = vectorizer.transform(X_train)
        self.model.fit(X_train_bow, y_train_encoded)

    def predict(self, X_test):
        """Predict dialog acts for test data."""
        X_test_bow = [self.transform_input(utterance) for utterance in X_test]
        X_test_bow = np.array(X_test_bow).reshape(len(X_test_bow), -1)
        return self.model.predict(X_test_bow)

    def predict_act(self, utterance):
        """Predict the dialog act of an utterance."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = vectorizer.transform([utterance])

        # Predict the dialog act using the trained logistic regression model
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

        return predicted_label[0] if len(predicted_label) > 0 else None

    def transform_input(self, utterance: str):
        """Transform the input utterance into a TF-IDF vector with OOV handling."""
        # Transform the utterance into a bag-of-words representation

        utterance_bow = TfidfVectorizer().transform([utterance])

        # Calculate the average TF-IDF vector for the words in the utterance
        if utterance_bow.nnz > 0:
            average_vector = utterance_bow.sum(axis=0) / utterance_bow.nnz
        else:
            # Handle the case where there are no words in the utterance (empty string)
            average_vector = np.zeros((1, len(vectorizer.get_feature_names_out())))

        return average_vector

class FeedForwardNeuralNetworkClassifier:
    """Feed-Forward Neural Network classifier for dialog act classification."""

    def __init__(self, name = "Feed-Forward Neural Network classifier"):
        self.name = name
        self.model = models.FeedForwardNeuralNetwork()
        self.oov_token = 0  # Special integer for out-of-vocabulary words

    def train(self, X_train, y_train):
        """Train the Feed-Forward Neural Network model and the label encoder."""
        y_train_encoded = label_encoder.transform(y_train)
        X_train_bow = vectorizer.transform(X_train)
        self.model.fit(X_train_bow, y_train_encoded)

    def predict(self, X_test):
        """Predict dialog acts for test data."""
        X_test = vectorizer.transform(X_test)
        return self.model.predict(X_test)

    def predict_act(self, utterance):
        """Predict the dialog act of an utterance."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = vectorizer.transform([utterance])

        # Predict the dialog act using the trained logistic regression model
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

        return predicted_label[0] if len(predicted_label) > 0 else None

    def transform_input(self, utterance: str):
        """Transform the input utterance into a TF-IDF vector with OOV handling."""
        # Transform the utterance into a bag-of-words representation

        utterance_bow = vectorizer.transform([utterance])

        # Calculate the average TF-IDF vector for the words in the utterance
        if utterance_bow.nnz > 0:
            average_vector = utterance_bow.sum(axis=0) / utterance_bow.nnz
        else:
            # Handle the case where there are no words in the utterance (empty string)
            average_vector = np.zeros((1, len(vectorizer.get_feature_names_out())))

        return average_vector