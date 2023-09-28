import classifiers
import utterance_extraction_recommendation as uer
import levenshtein_spellchecking as ls
import pandas as pd

class FiniteStateMachine:
    def __init__(self, restaurant_data: pd.DataFrame, configuration: dict, classifier=None, startstate: int=0, endstate: int=18):
        self._state = startstate
        self._start = startstate
        self._end = endstate
        self._configuration = configuration
        self._storedstring = ""  # String to output to user.
        if classifier == None or self._configuration['use_rulebaseline']:
            self._classifier = classifiers.RuleBaselineClassifier()  # The output of the function must be a dialog_act as in a string!
        else:
            self._classifier = classifier # classifier needs to be trained beforehand
        self._terminated = False
        self._restaurant_db = restaurant_data

        self._probable_food = ""  # Variables to hold extracted preferences until they can be confirmed.
        self._probable_area = ""
        self._probable_pricerange = ""

        self._preferred_food = ""  # Variables to hold confirmed extracted preferences.
        self._preferred_area = ""
        self._preferred_pricerange = ""

        self._probable_restaurant = ""
        self._preferred_restaurant = ""
        self._possible_recommendations = pd.DataFrame()

        self._represent_string_food = ""
        self._represent_string_area = ""
        self._represent_string_pricerange = ""

        self._additional_requirements = []
    
    def logic(self, inp: str):
        # One small problem with the current structure; currently the hello state system utterance will always be skipped, not sure if that's intentional.
        self.input_handler(inp)
        return self.output_handler()
    
    def input_handler(self, inp: str):
        """Manually defined function for handling logic with string inputs"""
        # inp = inp.lower()
        if self.get_state() == 1:  # Hello
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "hello":
                self.add_speech("Hello, human. I can help you find a restaurant based on your preferences. What kind of cuisine are you interested in?")
                self.set_state(2)
                return
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["food"] == "":  # No valid food detected.
                    self._preferred_food = uid["food"]
                    self.add_speech("I'm sorry, human. I could not understand what food type you preferred. Please state the type of cuisine you are interested in.")
                    self.set_state(2)
                    return
                if uid["food"] != "" and uid["food"] != ls.food_spellcheck(uid["food"], 3):  # Food likely misspelled
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["food"], ls.food_spellcheck(uid["food"], 3)))
                    self._probable_food = ls.food_spellcheck(uid["food"], 3)
                    self.set_state(3)
                    return
                else:
                    self._preferred_food = uid["food"]

                if uid["area"] == "":  # No valid area detected.
                    self._preferred_food = uid["food"]
                    self.add_speech("I did understand that you are interested in {} cuisine, but could not understand what area you preferred. Please state the area you're interested in.".format(self._preferred_food))
                    self.set_state(4)
                    return
                if uid["area"] != "" and uid["area"] != ls.area_spellcheck(uid["area"], 3):  # Area likely misspelled
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["area"], ls.area_spellcheck(uid["area"], 3)))
                    self._probable_area = ls.area_spellcheck(uid["area"], 3)
                    self.set_state(5)
                    return
                else:
                    self._preferred_area = uid["area"]

                if uid["pricerange"] == "":  # No valid price range detected.
                    self._preferred_food = uid["food"]
                    self.add_speech("I did understand you are interested in {} cuisine in the {} area, but could not understand what price range you preferred. Please state the price range you're looking for.".format(self._preferred_food, self._preferred_area))
                    self.set_state(6)
                    return
                if uid["pricerange"] != "" and uid["pricerange"] != ls.pricerange_spellcheck(uid["pricerange"], 3):  # Price range likely misspelled
                    self.add_speech("I could not find any price range to {}, are you perhaps interested in a {} price range?".format(uid["pricerange"], ls.pricerange_spellcheck(uid["pricerange"], 3)))
                    self._probable_pricerange = ls.pricerange_spellcheck(uid["pricerange"], 3)
                    self.set_state(7)
                    return
                else:
                    self._preferred_pricerange = uid["pricerange"]

                self.add_speech("Alright, I think I got all your preferences. Do you have any further requirements?")
                self.set_state(8)
                return
            else:
                self.add_speech("I did not detect an intent to supply information, human. Please try again.")
                self.set_state(1)
                return
        
        elif self.get_state() == 2:  # Ask for food preference
            # do something to extract food from utterance
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["food"] == "":  # No valid food detected.
                    self.add_speech("I'm sorry, human. I could not understand what food type you preferred. Please state the type of cuisine you are interested in.")
                    self.set_state(2)
                    return
                elif uid["food"] == "dontcare":
                    self._preferred_food = uid["food"]
                    self.add_speech("You have stated that you have no preference at all for the cuisine type of the restaurant, human. What area are you interested in?")
                    self.set_state(4)
                    return
                elif uid["food"] != ls.food_spellcheck(uid["food"], 3):
                    self.add_speech("I could not find any food related to {}, are you perhaps interested in {} cuisine?".format(uid["food"], ls.food_spellcheck(uid["food"], 3)))
                    self._probable_food = ls.food_spellcheck(uid["food"], 3)
                    self.set_state(3)
                    return
                elif uid["food"] != "" and uid["food"] == ls.food_spellcheck(uid["food"], 3):
                    self._preferred_food = uid["food"]
                    self.add_speech("I understood that you are interested in {} cuisine, what area are you looking for?".format(self._preferred_food))
                    self.set_state(4)
                    return
                else:
                    self.add_speech("Sorry, human. I did not understand what you meant. Can you please clarify what kind of food you are looking for?")
                    return
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Human, please state the type of food you are interested in. Or, if you want, we can start over.")
                return

        elif self.get_state() == 3:  # Food spelling check
            # check whether food preference was extracted properly from previous state
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_food = self._probable_food
                self._probable_food = ""
                self.add_speech("Very well, you are interested in {} cuisine. What area are you looking for?".format(self._preferred_food))
                self.set_state(4)
                return
            elif dialog_act == "negate" or dialog_act == "deny":
                self.add_speech("Okay. Would you like to state again what cuisine you are interested in?")
                self.set_state(2)
                return
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Is my recommendation correct human? Or would you perhaps like to start over?")
                return


        elif self.get_state() == 4:  # Ask for area preference
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["area"] == "":  # No valid area detected.
                    self.add_speech("I'm sorry, human. I could not understand what area you preferred. Please state the area in which you want to find a restaurant.")
                    self.set_state(4)
                    return
                if uid["area"] == "dontcare":  # No valid area detected.
                    self._preferred_area = uid["area"]
                    self.add_speech("You have stated that you have no preference at all for the restaurant's area, human. What price range are you interested in?".format(self._preferred_food))
                    self.set_state(6)
                    return
                elif uid["area"] != ls.area_spellcheck(uid["area"], 3):
                    self.add_speech("I could not find any area related to {}, are you perhaps interested in the {} area?".format(uid["area"], ls.area_spellcheck(uid["area"], 3)))
                    self._probable_area = ls.area_spellcheck(uid["area"], 3)
                    self.set_state(5)
                    return
                elif uid["area"] != "" and uid["area"] == ls.area_spellcheck(uid["area"], 3):
                    self._preferred_area = uid["area"]
                    self.add_speech("I understood that you are interested in restaurants in the {} area, what price range are you looking for?".format(uid["area"]))
                    self.set_state(6)
                    return
                else:
                    self.add_speech("Sorry, human. I did not understand what you meant. Would you clarify what kind of area you are interested in?")
                    return
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Human, please state the area you are interested in. Or, if you want, we can start over.")
                return

        elif self.get_state() == 5:  # Suggest spelling (area)
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_area = self._probable_area
                self._probable_area = ""
                self.add_speech("Very well, you are interested in a restaurant in the {} area. What price range are you interested in?".format(self._preferred_area))
                self.set_state(6)
                return
            elif dialog_act == "negate" or dialog_act == "deny":
                self.add_speech("Okay. Would you like to state again what area you are interested in?")
                self.set_state(4)
                return
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Is my recommendation correct human? Or would you perhaps like to start over completely?")
                return

        elif self.get_state() == 6:  # Ask for pricerange preference
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["pricerange"] == "":  # No valid area detected.
                    self.add_speech("I'm sorry, human. I could not understand what price range you preferred. Please state the price range of restaurants you are interested in.")
                    self.set_state(6)
                    return
                if uid["pricerange"] == "dontcare":
                    self._preferred_pricerange = uid["pricerange"]
                    self.add_speech("You have stated that you have no preference at all for the price of the restaurant, human. Do you have any further requirements?")
                    self.set_state(8)
                    return
                elif uid["pricerange"] != ls.pricerange_spellcheck(uid["pricerange"], 3):
                    self.add_speech("I could not find any {} price range, are you perhaps interested in a {} restaurant?".format(uid["pricerange"], ls.food_spellcheck(uid["pricerange"], 3)))
                    self.set_state(7)
                    return
                elif uid["pricerange"] != "" and uid["pricerange"] == ls.pricerange_spellcheck(uid["pricerange"], 3):
                    self._preferred_pricerange = uid["pricerange"]
                    self.add_speech("I understood that you are interested in a {} restaurant. Do you have any further requirements?".format(self._preferred_pricerange))
                    self.set_state(8)
                    return
                else:
                    self.add_speech("Sorry, human. I did not understand what you meant. Would you clarify what kind of price range you are looking for?")
                    return
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Human, please state the price range you are looking for. Or, if you want, we can start over.")
                return

        elif self.get_state() == 7:  # Spelling check (pricerange)
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_pricerange = self._probable_pricerange
                self._probable_pricerange = ""
                self.add_speech("Very well, I understood that you are interested in a {} restaurant. Do you have any further requirements?".format(self._preferred_pricerange))
                self.set_state(8)
                return
            elif dialog_act == "negate" or dialog_act == "deny":
                self._probable_pricerange = ""
                self.add_speech("Okay. Would you like to state again what price range you are interested in?")
                self.set_state(6)
                return
            elif dialog_act == "restart":
                self._probable_pricerange = ""
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Is my recommendation correct human? Or would you perhaps like to start over?")
                return
            
        elif self.get_state() == 8:  # Additional requests
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                # do something to extract additional information here
                pass
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
            elif dialog_act == "negate" or dialog_act == "deny":  # This is gonna be the only place where we mention *everything* now.
                # Format user preferences for string.
                if self._preferred_food == "dontcare":
                    self._represent_string_food = "any cuisine"
                else:
                    self._represent_string_food = self._preferred_food + " " + "cuisine"  # type: ignore
                if self._preferred_area == "dontcare":
                    self._represent_string_area = "any"
                else:
                    self._represent_string_area = "the" + " " + self._preferred_area  # type: ignore
                if self._preferred_pricerange == "dontcare":
                    self._represent_string_pricerange = "any price range"
                else:
                    self._represent_string_pricerange = self._preferred_pricerange + " " + "price range"  # type: ignore
                self.add_speech("Alright. I understand that you are interested in a {}, {} restaurant in {} area with the following requirements: {}. Is that correct?".format(self._represent_string_pricerange, self._represent_string_food, self._represent_string_area, self._additional_requirements))
                self.set_state(9)
                return
            else:
                self.add_speech("Human, do you have any (additional) requirements for the restaurant? Or would you like to start over?")

        elif self.get_state() == 9:  # Suggest restaurant
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                # do something to get restaurant.
                self._possible_recommendations = uer.provide_recommendations(self._restaurant_db, self._preferred_food, self._preferred_pricerange, self._preferred_area) # type: ignore
                if self._possible_recommendations.shape[0] >= 1:  # Found a restaurant!
                    self._probable_restaurant, self._possible_recommendations = uer.pop_recommendation(self._possible_recommendations)
                    self.add_speech(f"I have found a restaurant that matches your requirements human! It is the '{self._probable_restaurant}' restaurant. Would you like more information?")
                    self.set_state(10)
                    return
                else:  # Didn't find a restaurant
                    self.add_speech("I'm sorry, human. I did not find a restaurant which matches the given requirements. Would you like to start over or is my function fullfilled?")
                    self.set_state(11)
                    return
            elif dialog_act == "negate" or dialog_act == "deny":
                self.add_speech("Very well, let us start over, then. What kind of cuisine are you looking for?")
                self.set_state(2)
                return
            else:
                self.add_speech("Human, are you correctly looking for a {}, {} restaurant in {} area with the following requirements: {}?".format(self._represent_string_pricerange, self._represent_string_food, self._represent_string_area, self._additional_requirements))
                return

        elif self.get_state() == 10:  # Give information
            dialog_act = self.classifier_handler(inp)   
            if dialog_act == "reqmore":
                info_dict = uer.get_restaurant_info(restaurants_df=self._restaurant_db, restaurantname=self._probable_restaurant)  # type: ignore
                self.add_speech("Here's some information:")
                self.add_speech(f"Address: {info_dict['address']}")  # type: ignore
                self.add_speech(f"Phone number: {info_dict['phone']}")  # type: ignore
                self.add_speech(f"Zipcode: {info_dict['postcode']}")
                self.add_speech("Is there anything else I can be of assistance with? Can I perhaps provide the same information again, or is our conversation finished?")
                return
            elif dialog_act == "reqalts":
                self.add_speech("Not interested in this restaurant? Okay, let me see if I can find an alternative.")
                if len(self._possible_recommendations) < 1:
                    self.add_speech("I'm sorry, human. I did not find a restaurant which matches the given requirements. Would you like to start over or is my function fullfilled?")
                    self.set_state(11)
                    return
                self._probable_restaurant, self._possible_recommendations = uer.pop_recommendation(self._possible_recommendations)
                self.add_speech("I have found another restaurant that matches your requirements human! It is the '{}' restaurant. Would you like more information about this restaurant? Or am I done?".format(self._probable_restaurant))
                self.set_state(10)
                return
            else:
                self.add_speech("Human, would you like more information of the {} restaurant, or perhaps you want alternative restaurants?")
                return

        elif self.get_state() == 11:  # TODO: Pre-terminal state; possibility to revert to state 2 here.
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "bye" or dialog_act == "thankyou":
                self.add_speech("Alright human, I will now terminate. Did I perform amiably?")
                self.set_state(12)
                return
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)
                return
            else:
                self.add_speech("Human, is my function hereby fullfilled, or do you want to start over?")
                return

        elif self.get_state() == 12:  # Goodbye (terminate)
            self.add_speech("I am happy that I was able (or tried) to assist. Goodbye human.")
            self._terminated = True
    
    def classifier_handler(self, inp: str):
        return self._classifier.predict_act(inp)

    def add_speech(self, string: str):
        if self._configuration['output_all_caps']:
            string = string.upper()
        self._storedstring += string + "\n"

    def output_handler(self):
        tmp = self._storedstring
        self._storedstring = ""
        return tmp

    def get_state(self):
        return self._state
    
    def set_state(self, state: int):
        self._state = state
        if self._state == self._end:
            self._terminated = True
    
    def reset(self):
        self._state = self._start
        self._terminated = False
