import text_classification
import classifiers
import models
import fsm_statemanager as fsm
import utterance_extraction_recommendation as uer
import levenshtein_spellchecking as ls
import pandas as pd
import numpy as np

# global variables for configurability
spell_checking = True
ask_spell_checking_confirmation = True
output_all_caps = False
add_output_delay = False

# data imports
dialog_training_df, dialog_testing_df = text_classification.import_data(data_dir="Data/dialog_acts.dat", drop_duplicates=True)
restaurants_database = pd.read_csv("Data/restaurant_info.csv")

# add additional properties to restaurants_db
food_quality_vals = ['good food', 'mediocre food', 'bad food']
crowdedness_vals = ['busy', 'quiet']
length_stay_vals = ['long stay', 'medium stay', 'short stay']
restaurants_database['food_quality'] = np.random.choice(food_quality_vals, restaurants_database.shape[0])
restaurants_database['crowdedness'] = np.random.choice(crowdedness_vals, restaurants_database.shape[0])
restaurants_database['length_stay'] = np.random.choice(length_stay_vals, restaurants_database.shape[0])

# initialize and train dialog act classifier
classifier = classifiers.DialogActsClassifier()
classifier.train(x_train=dialog_training_df["utterance_content"], y_train=dialog_training_df["dialog_act"],)

# initialize dialog agent
manager = fsm.FiniteStateMachine(restaurant_data=restaurants_database, classifier=classifier, startstate=1)
while not manager._terminated:
    print(manager._state)
    print('>>>',end="")
    inp = input()
    out = manager.logic(inp)
    print(out)
