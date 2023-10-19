import random

import pandas as pd
import numpy as np
import time
import pickle

from src import text_classification
from src import classifiers
from src import fsm_statemanager as fsm

def ask_user_configurations():
    """Ask the user for their preferences regarding configurability of the dialog system. The preferences are saved in a dictionary which is returned after the function."""
    configurations = {}

    rulebaseline_preference = ['rulebaseline', 'Would you like to use the rule-based baseline model instead of the machine learning model? y/n\n', None]
    caps_preference = ['capitals', 'Would you like the system to reply in capital letters at all times? y/n\n', None]
    informal_preference = ['informal', 'Would you like the dialog system to reply with informal language instead of formal language? y/n\n', None]
    delay_preference = ['delay', 'Would you like the system to set a delay before it returns system utterances? Provide the number of seconds between 1 and 10, or "0" if you do not want any delay.\n', None]
    
    print('Please indicate how you would like the dialog system to be configured. There will be five questions in total.')
    for preference in [rulebaseline_preference, caps_preference, informal_preference]:
        while preference[2] not in ["y", "n"]:
            preference[2] = input(preference[1]).lower()
        if preference[2] == 'y':
            configurations[preference[0]] = True
        else:
            configurations[preference[0]] = False

    delays = ["0", "1", "2", "3", "4", "5", "6,", "7", "8", "9", "10"]
    while delay_preference[2] not in delays:
        delay_preference[2] = input(delay_preference[1])
    configurations[delay_preference[0]] = int(delay_preference[2])
    return configurations

def dialog_system(config: dict):
    """Initialize the dialog system."""
    # configuration values only used for testing, not experimenting.
    config['display_state_number'] = False

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

    if config['rulebaseline']:
        classifier = classifiers.RuleBaselineClassifier()
    else:
        # Load the pickled object from the file
        print('Loading classifier')
        with open('Data/FeedForwardsNeuralNetwork-deDuplicated.pkl', 'rb') as file:
            classifier = pickle.load(file)
        print('Classifier is ready.')

    # initialize dialog agent
    manager = fsm.FiniteStateMachine(restaurant_data=restaurants_database, configuration=config,
                                     classifier=classifier, startstate=1)
    # print welcome message
    if manager._configuration['informal']:
        greetstring = 'Hi there, welcome to this automated restaurant recommendation system! Let me know what you\'re looking for, and I will search some restaurants for you.'
        if manager._configuration['capitals']:
            print(greetstring.upper())
        else:
            print(greetstring)
    else:
        greetstring = 'Good day human! Welcome to this automated restaurant recommendation system. Please state what kind of restaurant you are looking for.'
        if manager._configuration['capitals']:
            print(greetstring.upper())
        else:
            print(greetstring)
    while not manager._terminated:
        if manager._configuration['display_state_number']:
            print(manager._state)
        inp = input('>>>')
        out = manager.logic(inp)
        if manager._configuration['delay'] > 0:
            delays = np.array([random.random(), random.random(), random.random(), random.random()])
            delays /= sum(delays)
            delays *= manager._configuration['delay']
            time.sleep(delays[0])
            print(".")
            time.sleep(delays[1])
            print("..")
            time.sleep(delays[2])
            print("...")
            time.sleep(delays[3])
        print(out.strip())

if __name__ == "__main__":
    #configuration = ask_user_configurations()

    # Define the URL and link text
    url = "https://forms.office.com/Pages/DesignPageV2.aspx?subpage=design&token=e5fc2499f31a45e6815ca9d5ebcf2649&id=oFgn10akD06gqkv5WkoQ5zmMNQ9l4eJJrWjubnAJsLxUOUZXT0NUTFBXNjhRRUFETENWNDAxTTNHRi4u"
    link_text = "Please click this link and fill out the form:"

    # Create the clickable link
    clickable_link = f'{link_text}\n{url}'
    # Print the clickable link
    print(clickable_link)
    input("Press the enter key to continue")


    #print(configuration)
    configurations = [[{'rulebaseline':True,'capitals':True,'informal':False,'delay':3},1],
     [{'rulebaseline':True,'capitals':True,'informal':False,'delay':0},2],
     [{'rulebaseline':True,'capitals':False,'informal':False,'delay':3},3],
     [{'rulebaseline':True,'capitals':False,'informal':False,'delay':0},4]]

    random.shuffle(configurations)

    for configuration in configurations:
        print("\n"*20)
        #print(f"This program version has the following settings")
        dialog_system(config=configuration[0])
        print(f"Please fill out the questionnaire section number {configuration[1]+1}"
              f" before continuing with the program")
        input("Press the enter key to continue")
