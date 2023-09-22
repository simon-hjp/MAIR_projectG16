import text_classification
import classifiers
import models
import fsm_statemanager as fsm
import utterance_extraction_recommendation as uer
import levenshtein_spellchecking as ls
import pandas as pd

# global variables here
spell_checking = True

# data imports
dialog_training_df, dialog_testing_df = text_classification.import_data(data_dir="Data/dialog_acts.dat", drop_duplicates=True)
restaurants_database = pd.read_csv("Data/restaurant_info.csv")

# initialize and train dialog act classifier
classifier = classifiers.DialogActsClassifier()
classifier.train(X_train=dialog_training_df["utterance_content"], y_train=dialog_training_df["dialog_act"],)

# initialize dialog agent
manager = fsm.FiniteStateMachine(restaurant_data=restaurants_database, classifier=classifier, startstate=1)
while not manager._terminated:
    print(manager._state)
    inp = input('>>>')
    manager.logic(inp)
