import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

###
##### Data importing and preprocessing
###

def import_data(data_dir: "/Users/berkeyazan/Desktop/dialog_acts.dat"):
    """Read data and return it as a dataframe.

    Parameters:
    - data_dir: {string} the directory where the .dat file is located.
    """
    # read data into DataFrame
    df = pd.read_table(data_dir, names=["Datapoint"])
    # to lowercase
    df["Datapoint"] = df["Datapoint"].str.lower()
    # split rows into labels and utterances
    df[["dialog_act", "utterance_content"]] = df["Datapoint"].str.split(
        pat=" ", n=1, expand=True
    )
    df.drop("Datapoint", axis=1, inplace=True)
    # create train- and test set
    df_train, df_test = train_test_split(df, test_size=0.15)
    return df_train, df_test


def transform_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Transform utterance dataframe into machine-readable bag-of-words representation."""
    labelencoder = LabelEncoder()
    labelencoder.fit(df_train["dialog_act"])
    labelencoder.transform(df_train["dialog_act"])
    labelencoder.transform(df_test["dialog_act"])
    # labelencoder

def bag_of_words(df_train, df_test):
    """Transform text data into bag-of-words representation using TF-IDF."""
    X_train = vectorizer.fit_transform(df_train['utterance_content'])
    X_test = vectorizer.transform(df_test['utterance_content'])
    return X_train, X_test




###
##### Baseline classifiers
###

class RuleBaselineClassifier():
    """Baseline model that classifies utterances based on keywords.
    """

    def utterance_contains_word(self, utterance, words):
        """Helper function to determine whether an utterance contains any word in a given list"""
        for word in words:
            if word in utterance:
                return True
        return False

    def predict_act(self, utterance) -> str:
        """Predict the dialog act of utterances using a set of rules."""
        if self.utterance_contains_word(utterance,
                                        ["food", "restaurant", "town", "east", "west", "south", "north", "part"]):
            return "inform"
        elif "it" in utterance and "is" in utterance:
            return "confirm"
        elif self.utterance_contains_word(utterance, ["yes", "right"]):
            return "affirm"
        elif self.utterance_contains_word(utterance, ["number", "phone", "address", "post"]):
            return "request"
        elif "thank" in utterance and "you" in utterance:
            return "thankyou"
        elif self.utterance_contains_word(utterance, ["noise", "sil", "unintelligible"]):
            return "null"
        elif self.utterance_contains_word(utterance, ["good", "bye"]):
            return "bye"
        elif ("how" in utterance and "about" in utterance) or "else" in utterance:
            return "reqalts"
        elif "no" in utterance:
            return "negate"
        elif self.utterance_contains_word(utterance, ["hi", "hello"]):
            return "hello"
        elif ("repeat" in utterance and "that" in utterance) or "repeat" in utterance:
            return "repeat"
        elif "okay" in utterance:
            return "ack"
        elif ("start" in utterance and "over" in utterance) or "start" in utterance:
            return "restart"
        elif self.utterance_contains_word(utterance, ["wrong", "dont"]):
            return "deny"
        elif "more" in utterance:
            return "reqmore"
        else:
            return "inform"  # Can replace this with 'error' later on if necessary for evaluation.


class MajorityBaselineClassifier:
    """Baseline model for text classification that classifies every input as the most common class
    in the dataset."""

    def __init__(self) -> None:
        self.prediction = ""

    def fit_training_data(self, df_train):
        """Find most common class and set it as the value to be predicted.

        Parameters
        - df_train: {pd.DataFrame} of shape (samples, features) containing dialog acts and utterances.

        """
        majority_class_label = df_train["dialog_act"].value_counts().idxmax()
        self.prediction = majority_class_label

    def predict_act(self, utterance):
        """Return the dialog act found to be most common during training.

        Parameters
        - utterance: {string} sentence to be classified. Only included for consistency with other classifiers.

        """
        return self.prediction


###
##### Machine learning classifiers
###


class LogisticRegressionClassifier:
    """Logistic Regression classifier for dialog act classification."""

    def __init__(self):
        self.model = LogisticRegression()
        self.label_encoder = LabelEncoder()
        self.oov_token = 0  # Special integer for out-of-vocabulary words

    def train(self, X_train, y_train):
        """Train the logistic regression model and the label encoder."""
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        self.model.fit(X_train, y_train_encoded)

    def predict(self, X_test):
        """Predict dialog acts for test data."""
        return self.model.predict(X_test)

    def predict_act(self, utterance):
        """Predict the dialog act of an utterance."""
        # Transform the utterance into a bag-of-words representation
        utterance_bow = vectorizer.transform([utterance])

        # Predict the dialog act using the trained logistic regression model
        predicted_label_encoded = self.model.predict(utterance_bow)

        # Decode the predicted label
        predicted_label = self.label_encoder.inverse_transform(predicted_label_encoded)

        return predicted_label[0] if len(predicted_label) > 0 else None

    def transform_input(self, utterance):
        """Transform the input utterance into a TF-IDF vector with OOV handling."""
        utterance_bow = vectorizer.transform([utterance])

        # Get the feature names from the vectorizer
        feature_names = vectorizer.get_feature_names_out()

        # Initialize a vector with zeros (OOV tokens)
        oov_vector = np.zeros(len(feature_names))

        # Process the TF-IDF vector to handle OOV words
        for word in utterance.split():
            if word in feature_names:
                word_index = feature_names.index(word)
                oov_vector[word_index] = utterance_bow[0, word_index]

        return oov_vector
def run():
    """Test each model, report performance, and then initiate the command-line for the user.
    """
    df_train, df_test = import_data("/Users/berkeyazan/Desktop/dialog_acts.dat")

    # majority baseline
    majority_model = MajorityBaselineClassifier()
    majority_model.fit_training_data(df_train)
    evaluate_model(majority_model, df_test)

    # rule-based baseline
    rule_model = RuleBaselineClassifier()
    evaluate_model(rule_model, df_test)

    # Bag of Words
    X_train_bow, X_test_bow = bag_of_words(df_train, df_test)

    # Logistic Regression
    lr_classifier = LogisticRegressionClassifier()
    lr_classifier.model = LogisticRegression(max_iter=100000)
    lr_classifier.train(X_train_bow, df_train['dialog_act'])
    evaluate_model(lr_classifier, df_test)

    user_choice = input(
        "Please specify which model you want to test:\n\
            A: majority class baseline\n\
            B: rule-based baseline\n\
            C: decision tree\n\
            D: logistic regression classifier\n"

    ).upper()
    model = None
    if user_choice == "A":
        model = majority_model
    elif user_choice == "B":
        model = rule_model
    elif user_choice == "C":
        print("Please choose another model, since this one has not been implemented\n")
    elif user_choice == "D":
        model = lr_classifier
    else:
        print("Please choose one of the listed options.\n")
        run()
    user_testing(model)
    return


def user_testing(model):
    """Predict an utterance given by the user with a trained model.

    Parameters:
    - model: {string} specify which model should be used.

    """
    while True:
        user_utterance = input(
            "Please provide the sentence the model has to classify. \nTo exit the program, enter '1'.\n"
        )
        if user_utterance == "1":
            return

        if model == "D":  # Logistic Regression classifier
            predicted_label = lr_classifier.predict([user_utterance])
            print(predicted_label[0])

        else:
            # For other models (Majority Baseline and Rule-Based Baseline)
            print(model.predict_act(user_utterance))


def evaluate_model(model, df_test):
    """Evaluate the performance of a trained model on the test dataset.

    Parameters:
    - model: a trained dialog act classifier
    - df_test: {pd.DataFrame} containing pairs of dialog acts and utterances.
    """
    pass


if __name__ == "__main__":
    run()