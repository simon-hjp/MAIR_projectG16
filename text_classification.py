import pandas as pd
from sklearn.model_selection import train_test_split


class RuleBaselineClassifier():
    def utterance_contains_word(self, utterance, words):
        """Helper function to determine whether an utterance contains any word in a given list"""
        for word in words:
            if word in utterance:
                return True
        return False

    def predict_act(self, utterances) -> list:
        """Predict the dialog act of utterances using a set of rules."""
        out = []
        for utterance in utterances:
            if self.utterance_contains_word(utterance, ["food", "restaurant", "town", "east", "west", "south", "north", "part"]):
            # if "any" in utterance or "food" in utterance or "restaurant" in utterance or "town" in utterance or "part" in utterance or "east" in utterance or "west" in utterance:
                out.append("inform")
            elif "it" in utterance and "is" in utterance:
                out.append("confirm")
            elif self.utterance_contains_word(utterance, ["yes", "right"]):
                out.append("affirm")
            elif self.utterance_contains_word(utterance, ["number", "phone", "address", "post"]):
                out.append("request")
            elif "thank" in utterance and "you" in utterance:
                out.append("thankyou")
            elif self.utterance_contains_word(utterance, ["noise", "sil", "unintelligible"]):
                out.append("null")
            # elif "good" in utterance or "bye" in utterance:
            elif self.utterance_contains_word(utterance, ["good", "bye"]):
                out.append("bye")
            elif ("how" in utterance and "about" in utterance) or "else" in utterance:
                out.append("reqalts")
            elif "no" in utterance:
                out.append("negate")
            elif self.utterance_contains_word(utterance, ["hi", "hello"]):
                out.append("hello")
            elif ("repeat" in utterance and "that" in utterance) or "repeat" in utterance:
                out.append("repeat")
            elif "okay" in utterance:
                out.append("ack")
            elif ("start" in utterance and "over" in utterance) or "start" in utterance:
                out.append("restart")
            elif self.utterance_contains_word(utterance, ["wrong", "dont"]):
                out.append("deny")
            elif "more" in utterance:
                out.append("reqmore")
            else:
                out.append("inform")  # Can replace this with 'error' later on if necessary for evaluation.
        return out

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


def run():
    """Test each model, report performance, and then initiate the command-line for the user.
    """
    df_train, df_test = import_data("dialog_acts.dat")
    # majority baseline
    majority_model = MajorityBaselineClassifier()
    majority_model.fit_training_data(df_train)
    # evaluate_model(majority_model, df_test)

    # rule-based baseline
    baseline_model = RuleBaselineClassifier()
    evaluate_model(baseline_model, df_test)

    # decision tree
    


    user_choice = input(
        "Please specify which model you want to test:\n\
            A: majority class baseline\n\
            B: rule-based baseline\n\
            C: machine-learning classifier 1\n\
            D: machine-learning classifier 2\n"
    ).upper()
    if user_choice == "A":
        model = majority_model
    elif user_choice == "B":
        model = RuleBaseline()
    elif user_choice == "C":
        print("Please choose another model, since this one has not been implemented\n")
        run()
    elif user_choice == "D":
        print("Please choose another model, since this one has not been implemented\n")
        run()
    else:
        print("Please choose one of the listed options.\n")
        run()
    user_testing(model)
    return


def import_data(data_dir: str):
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

def create_bags(df: pd.DataFrame):


def user_testing(model):
    """Predict an utterance given by the user with a trained model.

    Parameters:
    - model: {string} specify which model should be used.

    """
    continue_testing = True
    while continue_testing:
        user_utterance = input(
            "Please provide the sentence the model has to classify. \nTo exit the program, enter '1'.\n"
        )
        if user_utterance == "1":
            return
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
