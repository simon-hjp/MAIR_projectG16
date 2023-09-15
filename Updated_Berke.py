import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


###
##### Data importing and preprocessing
###


def import_data(data_dir: str):
    """Read data and return it as a dataframe.

    Parameters:
    - data_dir: {string} the directory where the .dat file is located.
    """
    # read data into DataFrame
    df = pd.read_table("/Users/berkeyazan/Desktop/dialog_acts.dat", names=["Datapoint"])
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


# def transform_data(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # """Transform utterance dataframe into machine-readable bag-of-words representation."""
    # labelencoder = LabelEncoder()
    # labelencoder.fit(df_train["dialog_act"])
    # labelencoder.transform(df_train["dialog_act"])
    # labelencoder.transform(df_test["dialog_act"])


def bag_of_words(df_train, df_test):
    """Transform text data into bag-of-words representation using TF-IDF."""
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df_train['utterance_content'])
    X_test = vectorizer.transform(df_test['utterance_content'])
    return X_train, X_test


###
##### Baseline classifiers
###


class MajorityBaselineClassifier:
    """Baseline model for text classification that classifies every input as the most common class
    in the dataset."""

    def __init__(self) -> None:
        self.name = "Majority baseline classifier"
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

    def predict(self, X_test: pd.DataFrame):
        """Return the prediction for each element in a test set."""
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
        if self.utterance_contains_word(
                utterance,
                ["food", "restaurant", "town", "east", "west", "south", "north", "part"],
        ):
            return "inform"
        elif "it" in utterance and "is" in utterance:
            return "confirm"
        elif self.utterance_contains_word(utterance, ["yes", "right"]):
            return "affirm"
        elif self.utterance_contains_word(
                utterance, ["number", "phone", "address", "post"]
        ):
            return "request"
        elif "thank" in utterance and "you" in utterance:
            return "thankyou"
        elif self.utterance_contains_word(
                utterance, ["noise", "sil", "unintelligible"]
        ):
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

    def predict(self, X_test: pd.DataFrame):
        """Predict every utterance in a given dataframe of features."""
        return [self.predict_act(element) for element in X_test]


###
##### Machine learning classifiers
###


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
        utterance_bow = self.vectorizer.transform([utterance])

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


def user_testing(model):
    """Predict an utterance given by the user with a trained model.

    Parameters:
    - model: model to be used.
    """
    while True:
        user_utterance = input(
            "Please provide the sentence the model has to classify. \nTo exit the program, enter '1'.\n"
        )
        if user_utterance == "1":
            return
        if model.name == "Logistic regression classifier":
            print(
                model.predict_act(user_utterance)
            )  # logistic regression requires a string, not a list
        else:
            print(model.predict_act(user_utterance))


def evaluate_model(model, df_test):
    """Evaluate the performance of a trained model on the test dataset.

    Parameters:
    - model: a trained dialog act classifier
    - df_test: {pd.DataFrame} containing pairs of dialog acts and utterances.
    """
    X_test = df_test["utterance_content"]
    y_test = df_test["dialog_act"]

    if hasattr(model, 'label_encoder'):
        # Encode the ground truth labels using the model's label encoder
        y_test_encoded = model.label_encoder.transform(y_test)
    else:
        y_test_encoded = y_test  # No label encoding needed

    y_hat = model.predict(X_test)

    print(f"{model.name} performance evaluation")
    print("\tAccuracy score:", accuracy_score(y_test_encoded, y_hat))
    print(
        "\tRecall score:", recall_score(y_test_encoded, y_hat, average="macro", zero_division=0)
    )
    print(
        "\tPrecision score:",
        precision_score(y_test_encoded, y_hat, average="macro", zero_division=0),
    )
    print("\tF1 score:", f1_score(y_test_encoded, y_hat, average="macro"))


###
##### Program control flow
###


def run():
    """Test each model, report performance, and then initiate the command-line for the user."""
    df_train, df_test = import_data("dialog_acts.dat")

    # majority baseline
    majority_model = MajorityBaselineClassifier()
    majority_model.fit_training_data(df_train)
    evaluate_model(majority_model, df_test)

    # rule-based baseline
    rule_model = RuleBaselineClassifier()
    evaluate_model(rule_model, df_test)

    # transform utterances to BOW
    X_train_bow, X_test_bow = bag_of_words(df_train, df_test)

    # decision tree

    # logistic regression
    lr_classifier = LogisticRegressionClassifier()
    lr_classifier.model = LogisticRegression(max_iter=10000)
    lr_classifier.train(df_train["utterance_content"], df_train["dialog_act"])
    evaluate_model(lr_classifier, df_test)

    model = None
    ask_input = True
    while ask_input:
        user_choice = input(
            "Please specify which model you want to test:\n\
        A: majority class baseline\n\
        B: rule-based baseline\n\
        C: machine-learning classifier 1\n\
        D: logistic regression\n"
        ).upper()
        if user_choice == "A":
            model = majority_model
            ask_input = False
        elif user_choice == "B":
            model = rule_model
            ask_input = False
        elif user_choice == "C":
            print(
                "Please choose another model, since this one has not been implemented\n"
            )
            continue
        elif user_choice == "D":
            model = lr_classifier
            ask_input = False
        else:
            print("Please choose one of the listed options.\n")
            continue
    user_testing(model)


if __name__ == "__main__":
    run()
