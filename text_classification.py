import pandas as pd
from sklearn.model_selection import train_test_split

# "" in utterance
def utterance_contains_word(utterance, words):
    """Helper function to determine whether an utterance contains any word in a given list"""
    for word in words:
        if word in utterance:
            return True
    return False

def baseline_keyword_algorithm(utterances) -> list:
    """Predict the dialog act of utterances using a set of rules."""
    out = []
    for utterance in utterances:
        if utterance_contains_word(utterance, ["food", "restaurant", "town", "east", "west", "south", "north", "part"]):
        # if "any" in utterance or "food" in utterance or "restaurant" in utterance or "town" in utterance or "part" in utterance or "east" in utterance or "west" in utterance:
            out.append("inform")
        elif "it" in utterance and "is" in utterance:
            out.append("confirm")
        elif utterance_contains_word(utterance, ["yes", "right"]):
            out.append("affirm")
        elif utterance_contains_word(utterance, ["number", "phone", "address", "post"]):
            out.append("request")
        elif "thank" in utterance and "you" in utterance:
            out.append("thankyou")
        elif utterance_contains_word(utterance, ["noise", "sil", "unintelligible"]):
            out.append("null")
        # elif "good" in utterance or "bye" in utterance:
        elif utterance_contains_word(utterance, ["good", "bye"]):
            out.append("bye")
        elif ("how" in utterance and "about" in utterance) or "else" in utterance:
            out.append("reqalts")
        elif "no" in utterance:
            out.append("negate")
        elif utterance_contains_word(utterance, ["hi", "hello"]):
            out.append("hello")
        elif ("repeat" in utterance and "that" in utterance) or "repeat" in utterance:
            out.append("repeat")
        elif "okay" in utterance:
            out.append("ack")
        elif ("start" in utterance and "over" in utterance) or "start" in utterance:
            out.append("restart")
        elif utterance_contains_word(utterance, ["wrong", "dont"]):
            out.append("deny")
        elif "more" in utterance:
            out.append("reqmore")
        else:
            out.append("inform")  # Can replace this with 'error' later on if necessary for evaluation.
    return out

class MajorityBaseline:
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
    """Initiate the command-line for the user.
    """
    df_train, df_test = import_data(
        "C:\\Users\\spals\\AI\\Master\\MAIR\\Data\\dialog_acts.dat"
    )
    majority_model = MajorityBaseline()
    majority_model.fit_training_data(df_train)
    # evaluate_model(majority_model, df_test)
    user_choice = input(
        "Please specify which model you want to test:\n\
            A: majority class baseline\n\
            B: rule-based baseline\n\
            C: machine-learning classifier 1\n\
            D: machine-learning classifier 2\n"
    ).upper()
    if user_choice == "A":
        user_testing(model="majority")
    elif user_choice == "B":
        user_testing(model="rules")
    elif user_choice == "C":
        pass
    elif user_choice == "D":
        pass
    else:
        print("Please choose one of the listed options.\n")
        run()
    evaluate_model(model, df_test)
    user_testing(model)


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


def user_testing(model):
    """Predict an utterance given by the user with a trained model.

    Parameters:
    - model: {string} specify which model should be used.

    """
    continue_testing = True
    while continue_testing:
        user_utterance = input(
            "Please provide the sentence the model has to classify. To exit the program, enter '1'.\n"
        )
        if user_utterance == "1":
            return
        print(model.predict_act(user_utterance))
    return


def evaluate_model(model, df_test):
    """Evaluate the performance of a trained model on the test dataset.

    Parameters:
    - model: a trained dialog act classifier
    - df_test: {pd.DataFrame} containing pairs of dialog acts and utterances.
    """
    pass


run()
