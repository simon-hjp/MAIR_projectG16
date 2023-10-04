import pandas as pd
import numpy as np
import time
import pickle

from src import text_classification
from src import classifiers
from src import fsm_statemanager as fsm

def dialog_system():
    # Dictionary containing the configuration values
    configuration_dict = {
        'use_rulebaseline': True,
        'output_all_caps': False,
        'add_output_delay': 0,
        'informal_switch': True,
        'display_state_number': False  # only used for testing, not experimenting.
    }

    # data imports
    dialog_training_df, dialog_testing_df = text_classification.import_data(data_dir="Data/dialog_acts.dat")
    restaurants_database = pd.read_csv("Data/restaurant_info.csv")

    # add additional properties to restaurants_db
    food_quality_vals = ['good food', 'bad food']
    crowdedness_vals = ['busy', 'quiet']
    length_stay_vals = ['long stay', 'short stay']
    restaurants_database['food_quality'] = np.random.choice(food_quality_vals, restaurants_database.shape[0])
    restaurants_database['crowdedness'] = np.random.choice(crowdedness_vals, restaurants_database.shape[0])
    restaurants_database['length_stay'] = np.random.choice(length_stay_vals, restaurants_database.shape[0])

    # initialize and train dialog act classifier
    classifiers.label_encoder.fit(dialog_training_df["dialog_act"])
    classifiers.vectorizer.fit(dialog_training_df["utterance_content"])

    if configuration_dict['use_rulebaseline']:
        classifier = classifiers.RuleBaselineClassifier()
    else:
        # Load the pickled object from the file
        print('Loading classifier')
        with open('FeedForwardsNeuralNetwork-deDuplicated.pkl', 'rb') as file:
            classifier = pickle.load(file)
        print('Classifier is ready.')

    # initialize dialog agent
    manager = fsm.FiniteStateMachine(restaurant_data=restaurants_database, configuration=configuration_dict,
                                     classifier=classifier, startstate=1)
    # print welcome message
    if manager._configuration['informal_switch']:
        print(
            'Hi there, welcome to this automated restaurant recommendation system! Let me know what you\'re looking for, and I will search some restaurants for you.')
    else:
        print(
            'Good day human! Welcome to this automated restaurant recommendation system. Please state what kind of restaurant you are looking for.')
    while not manager._terminated:
        if manager._configuration['display_state_number']:
            print(manager._state)
        inp = input('>>>')
        out = manager.logic(inp)
        if manager._configuration['add_output_delay']:
            time.sleep(manager._configuration['add_output_delay'])
        print(out)


if __name__ == "__main__":
    dialog_system()
