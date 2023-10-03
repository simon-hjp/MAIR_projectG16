####
#     Methods in AI research: part 1b
#     Group 16
####

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

### our code

import models
import classifiers as cl


###
# Data importing and initial preprocessing
###

def import_data(data_dir: str, drop_duplicates=False):
    """Read data and return it as a dataframe.
    """
    df = pd.read_table(data_dir, names=["Datapoint"])
    df["Datapoint"] = df["Datapoint"].str.lower()

    # split rows into labels and utterances
    df[["dialog_act", "utterance_content"]] = df["Datapoint"].str.split(pat=" ", n=1, expand=True)
    df.drop("Datapoint", axis=1, inplace=True)

    if drop_duplicates:
        df = df.drop_duplicates(keep="first", inplace=False, ignore_index=False)

    # create train- and test set
    df_train, df_test = train_test_split(df, test_size=0.15)
    return df_train, df_test


###
# Evaluation
###

def evaluate_model(model, df_test):
    """Evaluate the performance of a trained model on the test dataset.
    """
    x_test = df_test["utterance_content"]
    y_test = df_test["dialog_act"]

    y_hat = model.predict(x_test)

    if not (model.name == "Majority baseline classifier" or model.name == "Rule-based baseline classifier"):
        y_hat = cl.label_encoder.inverse_transform(y_hat)
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
            "Please provide the sentence the model has to classify. \nTo go back, enter '1'.\n>>"
        ).lower()
        if user_utterance == "1":
            break
        print(f'Your utterance was "{user_utterance}"')
        print(f"The {model.name} guessed: {model.predict_act(user_utterance)}\n")


###
##### Program control flow
###

def command_line_testing(models_dict: dict):
    """Let a user specify a model which they can then test."""
    # create a string to show the model options
    options = ''.join([key + ": " + models_dict[key].name + "\n\t" for key in models_dict])
    # options = ''.join(options)
    # let the user choose a model
    while True:
        user_choice = input(
            f"Please specify which model you want to test:\n\t{options}1: quit the program.\n>>").upper()
        if user_choice in models_dict:
            user_testing(models_dict[user_choice])
            continue
        if user_choice == '1':
            return
        else:
            print("Wrong choice, choose another option.")
            continue


def run(data_dir="Data\\dialog_acts.dat"):
    """Train models and report their performance, then initiate the command line testing."""

    trained_models = models.create_models(data_dir=data_dir)
    command_line_testing(trained_models)


if __name__ == "__main__":
    run(data_dir="dialog_acts.dat")
