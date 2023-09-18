######
###     Methods in AI research: part 1a
###     Group 16
######

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

###

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

###
## Data importing and initial preprocessing
###


def import_data(data_dir: str, drop_duplicates=False):
    """Read data and return it as a dataframe.
    """
    df = pd.read_table(data_dir, names=["Datapoint"])
    df["Datapoint"] = df["Datapoint"].str.lower()

    # split rows into labels and utterances
    df[["dialog_act", "utterance_content"]] = df["Datapoint"].str.split(
        pat=" ", n=1, expand=True
    )
    df.drop("Datapoint", axis=1, inplace=True)

    if drop_duplicates:
        df = df.drop_duplicates(keep="first", inplace=False, ignore_index=False)

    # create train- and test set
    df_train, df_test = train_test_split(df, test_size=0.15)
    return df_train, df_test


###
##### Baseline classifiers
###


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


###
##### Machine learning classifiers
###


class DialogActsClassifier:
    """Machine learning classifier for dialog act classification."""

    def __init__(self, model=None, name="Decision tree classifier"):
        if model is None:
            self.model = DecisionTreeClassifier()
        else:
            self.model = model
        self.name = name
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()

    def train(self, X_train, y_train, hyperparams_dict=None):
        """Fit the label encoder, the bag-of-words vectorizer and train the model."""
        # fit label encoder and transform dependent variable
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        
        # fit bag-of-words vectorizer and transform features
        self.vectorizer.fit(X_train)
        X_train_bow = self.vectorizer.transform(X_train)
        
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
        utterance_bow = self.vectorizer.transform([utterance])
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = self.label_encoder.inverse_transform(predicted_label_encoded)

        return predicted_label[0] if len(predicted_label) > 0 else None

    def transform_input(self, utterance: str):
        """Transform the input utterance into a TF-IDF vector with OOV handling."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = self.vectorizer.transform([utterance])

        # Calculate the average TF-IDF vector for the words in the utterance
        if utterance_bow.nnz > 0:
            average_vector = utterance_bow.sum(axis=0) / utterance_bow.nnz
        else:
            # Handle the case where there are no words in the utterance (empty string)
            average_vector = np.zeros((1, len(self.vectorizer.get_feature_names_out())))

        return average_vector

class LogisticRegressionClassifier:
    """Logistic Regression classifier for dialog act classification."""

    def __init__(self):
        self.name = "Logistic regression classifier"
        self.model = LogisticRegression()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.oov_token = 0  # Special integer for out-of-vocabulary words

    def train(self, X_train, y_train):
        """Train the logistic regression model and the label encoder."""
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        self.vectorizer.fit(X_train)
        X_train_bow = self.vectorizer.transform(X_train)
        self.model.fit(X_train_bow, y_train_encoded)

    def predict(self, X_test):
        """Predict dialog acts for test data."""
        X_test_bow = [self.transform_input(utterance) for utterance in X_test]
        X_test_bow = np.array(X_test_bow).reshape(len(X_test_bow), -1)
        return self.model.predict(X_test_bow)

    def predict_act(self, utterance):
        """Predict the dialog act of an utterance."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = self.vectorizer.transform([utterance])

        # Predict the dialog act using the trained logistic regression model
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = self.label_encoder.inverse_transform(predicted_label_encoded)

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
            average_vector = np.zeros((1, len(self.vectorizer.get_feature_names_out())))

        return average_vector


class FeedForwardNeuralNetwork():
    def __init__(self, name = "Feed-Forward Neural Network model"):
        self.name = name
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.oov_token = 0  # Special integer for out-of-vocabulary words

        self.model = keras.Sequential(
            [
                layers.Dense(20, activation="relu", name="layer1"),
                layers.Dense(10, activation="tanh", name="layer4"),
                layers.Dense(1, activation='softmax', name='custom_output_layer')
            ]
        )
        self.loss_fn = keras.losses.CategoricalCrossentropy()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.1)

        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer)
    def fit(self,x_train, y_train):
        # Training loop

        # Batch size for each training iteration
        batch_size = 1

        # find the size of the training data
        training_length = x_train.shape[0]
        # boolean to build the model on the first run
        first = True

        # change the csr_matrix into a regular matrix for our applications, and reshape it at the same time
        x_train = [x_train[x, :].toarray().reshape(1, -1) for x in range(training_length)]

        for batch_start in range(0, training_length, batch_size):
            # current batch to train on
            batch_x = x_train[batch_start: batch_start + batch_size]
            batch_x = batch_x[0]
            batch_y = y_train[batch_start: batch_start + batch_size]
            if len(batch_x) < batch_size:
                # if the batch is too big for the remainder of the training set, the training batch
                # size does not handle modularity
                continue

            if first:
                # build the neural-net on the batch ( this is not training)
                self.model(batch_x)
                print(self.model.summary())
                first = False

            # Compute gradients and update weights
            with tf.GradientTape() as tape:
                # this prediction is not doing anything
                predictions = self.model(batch_x)
                #-/-|-\-#
                # attempt to reshape the y so that the datastructure is identical for the loss_value function
                batch_y = tf.constant(batch_y)
                batch_y = tf.transpose(batch_y)
                batch_y = tf.expand_dims(batch_y, axis=1)
                #-/-|-\-#
                loss_value = self.loss_fn(batch_y, predictions[0])
            # changes to parameters
            gradients = tape.gradient(loss_value, self.model.trainable_weights)
            # apply the changes
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            # to see how the model is adapting, spaces are nice
            # print("\n"*25)
            print(f"run number:{batch_start}, loss_val:{loss_value}\nprediction:{predictions}, correct:{batch_y}")
        # used for inspecting the code with breakpoints
        pass

    def predict(self, sentence):
        return self.model(sentence)
class FeedForwardNeuralNetworkClassifier:
    """Logistic Regression classifier for dialog act classification."""

    def __init__(self, name = "Feed-Forward Neural Network classifier"):
        self.name = name
        self.model = FeedForwardNeuralNetwork()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        self.oov_token = 0  # Special integer for out-of-vocabulary words

    def train(self, X_train, y_train):
        """Train the Feed-Forward Neural Network model and the label encoder."""
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        self.vectorizer.fit(X_train)
        X_train_bow = self.vectorizer.transform(X_train)
        self.model.fit(X_train_bow, y_train_encoded)

    def predict(self, X_test):
        """Predict dialog acts for test data."""
        X_test_bow = [self.transform_input(utterance) for utterance in X_test]
        X_test_bow = np.array(X_test_bow).reshape(len(X_test_bow), -1)
        return self.model.predict(X_test_bow)

    def predict_act(self, utterance):
        """Predict the dialog act of an utterance."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = self.vectorizer.transform([utterance])

        # Predict the dialog act using the trained logistic regression model
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = self.label_encoder.inverse_transform(predicted_label_encoded)

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
            average_vector = np.zeros((1, len(self.vectorizer.get_feature_names_out())))

        return average_vector
###
##### Evaluation
###


def evaluate_model(model, df_test):
    """Evaluate the performance of a trained model on the test dataset.
    """
    X_test = df_test["utterance_content"]
    y_test = df_test["dialog_act"]

    y_hat = model.predict(X_test)

    if hasattr(model, "label_encoder"):
        y_hat = model.label_encoder.inverse_transform(y_hat)

    # calculate performance metrics
    print(f"{model.name} performance evaluation")
    model_performance = classification_report(y_test, y_hat, output_dict=True, zero_division=0)

    # organize performance metrics in a dataframe
    model_performance = pd.DataFrame(model_performance).transpose()
    model_performance.drop("support", axis=1, inplace=True)
    model_performance.index.name = "Dialog act"
    print(model_performance)


def user_testing(model):
    """Predict an utterance given by the user with a trained model.
    """
    print(f"Using dialog act classifier: {model.name}.\n")
    while True:
        # user input sentence converted to lower case to prevent errors
        user_utterance = input(
            "Please provide the sentence the model has to classify. \nTo exit the program, enter '1'.\n>>"
        ).lower()
        if user_utterance == "1":
            break
        print(f'Your utterance was "{user_utterance}"')
        print(f"The {model.name} guessed: {model.predict_act(user_utterance)}\n")


###
##### Program control flow
###


def run(data_dir="dialog_acts.dat"):
    """Test each model, report performance, and then initiate the command-line for the user."""

    # Use this boolean to drop duplicate values from the data source
    df_train, df_test = import_data(data_dir=data_dir, drop_duplicates=False)
    df_train_deduplicated, df_test_deduplicated = import_data(data_dir=data_dir, drop_duplicates=True)

    """
    # majority baseline
    majority_model = MajorityBaselineClassifier()
    majority_model.train(df_train)
    evaluate_model(majority_model, df_test)

    # rule-based baseline
    rule_model = RuleBaselineClassifier()
    evaluate_model(rule_model, df_test)

    # decision tree
    dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }

    dt_classifier = DialogActsClassifier(name="Decision tree classifier")
    dt_classifier.train(
        df_train["utterance_content"],
        df_train["dialog_act"],
        hyperparams_dict=dt_params,
    )
    evaluate_model(dt_classifier, df_test)

    # decision tree without duplicates
    dt_classifier_dd = DialogActsClassifier(name="Decision tree without duplicates")
    dt_classifier_dd.train(
        df_train_deduplicated["utterance_content"],
        df_train_deduplicated["dialog_act"],
        hyperparams_dict=dt_params,
    )
    evaluate_model(dt_classifier_dd, df_test_deduplicated)


    # logistic regression
    lr_classifier = DialogActsClassifier( model=LogisticRegression(max_iter=10000), name="Logistic regression")
    lr_classifier.train(df_train["utterance_content"], df_train["dialog_act"])
    evaluate_model(lr_classifier, df_test)

    # logistic regression without duplicates
    lr_classifier_dd = DialogActsClassifier(
        model=LogisticRegression(max_iter=10000),
        name="Logistic regression without duplicates",
    )
    lr_classifier_dd.train(
        df_train_deduplicated["utterance_content"], df_train_deduplicated["dialog_act"]
    )
    evaluate_model(lr_classifier_dd, df_test_deduplicated)
    """
    # Feed-Forward Neural Network
    ffnn_classifier = FeedForwardNeuralNetworkClassifier(name="Feed-Forward Neural Network")
    ffnn_classifier.train(df_train["utterance_content"], df_train["dialog_act"])
    evaluate_model(ffnn_classifier, df_test)

    # Feed-Forward Neural Network without duplicates
    ffnn_classifier_dd = FeedForwardNeuralNetworkClassifier(name="Feed-Forward Neural Network without duplicates")
    ffnn_classifier_dd.train(df_train_deduplicated["utterance_content"], df_train_deduplicated["dialog_act"])
    evaluate_model(ffnn_classifier_dd, df_test_deduplicated)

    model = None
    while True:
        user_choice = input(
            "Please specify which model you want to test:\n\
        A: majority class baseline\n\
        B: rule-based baseline\n\
        C: decision tree classifier\n\
        D: decision tree classifier with deduplicated data\n\
        E: logistic regression classifier\n\
        F: logistic regression classifier with deduplicated data\n\
        1: quit the program.\n"
        ).upper()
        if user_choice == "A":
            model = majority_model
        elif user_choice == "B":
            model = rule_model
        elif user_choice == "C":
            model = dt_classifier
        elif user_choice == "D":
            model = dt_classifier_dd
        elif user_choice == "E":
            model = lr_classifier
        elif user_choice == "F":
            model = lr_classifier_dd
        elif user_choice == "1":
            return
        else:
            print("Wrong choice, choose another option.")
            continue
        break
    user_testing(model)


if __name__ == "__main__":
    run(data_dir="dialog_acts.dat")
