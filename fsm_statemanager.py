import classifiers
import utterance_extraction_recommendation as uer
import levenshtein_spellchecking as ls
import pandas as pd

class FiniteStateMachine:
    def __init__(self, restaurant_data: pd.DataFrame, classifier=None, startstate: int=0, endstate: int=18):
        self._state = startstate
        self._start = startstate
        self._end = endstate
        self._storedstring = ""  # String to output to user.
        if classifier == None:
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
    
    def logic(self, inp: str):
        # One small problem with the current structure; currently the hello state system utterance will always be skipped, not sure if that's intentional.
        self.input_handler(inp)
        return self.output_handler()
    
    def input_handler(self, inp: str):
        """Manually defined function for handling logic with string inputs"""
        if self.get_state() == 1:  # Hello
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "hi":
                self.add_speech("Hello, human. I can help you find a restaurant based on your preferences. What kind of cuisine are you interested in?")
                self.set_state(2)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary

                if uid["food"] == "":  # No valid food detected.
                    self.add_speech("I'm sorry, human. I could not understand what food type you preferred. Please state the type of cuisine you are interested in.")
                    self.set_state(2)
                elif uid["food"] != ls.food_spellcheck(uid["food"], 3):  # Food likely misspelled
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["food"], ls.food_spellcheck(uid["food"], 3)))
                    self._probable_food = ls.food_spellcheck(uid["food"], 3)
                    self.set_state(3)
                else:
                    self._preferred_food = uid["food"]

                if uid["area"] == "":  # No valid area detected.
                    self.add_speech("I did understand that you are interested in {} cuisine, but could not understand what area you preferred. Please state the area you're interested in.".format(self._preferred_food))
                    self.set_state(4)
                elif uid["area"] != ls.area_spellcheck(uid["area"], 3):  # Area likely misspelled
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["area"], ls.area_spellcheck(uid["area"], 3)))
                    self._probable_area = ls.area_spellcheck(uid["area"], 3)
                    self.set_state(5)
                else:
                    self._preferred_area = uid["area"]

                if uid["pricerange"] == "":  # No valid price range detected.
                    self.add_speech("I did understand you are interested in {} cuisine in the {} area, but could not understand what price range you preferred. Please state the price range you're looking for.".format(self._preferred_food, self._preferred_area))
                    self.set_state(6)
                elif uid["pricerange"] != ls.pricerange_spellcheck(uid["pricerange"], 3):  # Price range likely misspelled
                    self.add_speech("I could not find any price range to {}, are you perhaps interested in a {} price range?".format(uid["pricerange"], ls.pricerange_spellcheck(uid["pricerange"], 3)))
                    self._probable_pricerange = ls.pricerange_spellcheck(uid["pricerange"], 3)
                    self.set_state(7)
                else:
                    self._preferred_pricerange = uid["pricerange"]

                self.add_speech("Alright, I understood you prefer a {} restaurant serving {} food, in the {} area.".format(self._preferred_pricerange,self._preferred_food, self._preferred_area))
                self.set_state(8)
            else:
                self.add_speech("I did not detect an intent to supply information, human. Please try again.")
                self.set_state(1)
        
        elif self.get_state() == 2:  # Ask for food preference
            # do something to extract food from utterance
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["food"] == "":
                    self.add_speech("I'm sorry, human. I could not understand what food type you preferred. Please try again.")
                    self.set_state(2)
                elif uid["food"] != ls.food_spellcheck(uid["food"], 3):
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["food"], ls.food_spellcheck(uid["food"], 3)))
                    self._probable_food = ls.food_spellcheck(uid["food"], 3)
                    self.set_state(3)
                elif uid["food"] != "" and uid["food"] == ls.food_spellcheck(uid["food"], 3):
                    self._preferred_food = uid["food"]
                    self.add_speech("I understood that you are interested in {} cuisine, what area are you looking for?".format(self._preferred_food))
                    self.set_state(4)

        elif self.get_state() == 3:  # Food spelling check
            # check whether food preference was extracted properly from previous state
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_food = self._probable_food
                self._probable_food = ""
                self.add_speech("Very well, you are interested in {} cuisine. What area are you looking for?".format(self._preferred_food))
                self.set_state(4)
            elif dialog_act == "negate" or dialog_act == "deny":
                self.add_speech("Okay. Would you like to state again what cuisine you are interested in?")
                self.set_state(2)
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)


        elif self.get_state() == 4:  # Ask for area preference
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["area"] == "":
                    self.add_speech("I'm sorry, human. I could not understand what area you want. Please try again.")
                    self.set_state(4)
                elif uid["area"] != ls.area_spellcheck(uid["area"], 3):
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["area"], ls.area_spellcheck(uid["area"], 3)))
                    self._probable_area = ls.area_spellcheck(uid["area"], 3)
                    self.set_state(5)
                elif uid["area"] != "" and uid["area"] == ls.area_spellcheck(uid["rea"], 3):
                    self._preferred_area = uid["area"]
                    self.add_speech("I understood that you are interested in restaurants in the {} area, what price range are you looking for?".format(uid["area"]))
                    self.set_state(6)

        elif self.get_state() == 5:  # Suggest spelling (area)
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_area = self._probable_area
                self._probable_area = ""
                self.add_speech("Very well, you are interested in a restaurant in the {} area. What price range are you interested in?".format(self._preferred_area))
                self.set_state(6)
            elif dialog_act == "negate" or dialog_act == "deny":
                self.add_speech("Okay. Would you like to state again what area you are interested in?")
                self.set_state(4)
            elif dialog_act == "restart":
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)

        elif self.get_state() == 6:  # Ask for pricerange preference
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary
                if uid["pricerange"] == "":
                    self.add_speech("I'm sorry, human. I could not understand what prince range type you are interested. Please try again.")
                    self.set_state(6)
                elif uid["pricerange"] != ls.pricerange_spellcheck(uid["pricerange"], 3):
                    self.add_speech("I could not find any {} price range, are you perhaps interested in a {} restaurant?".format(uid["pricerange"], ls.food_spellcheck(uid["pricerange"], 3)))
                    self.set_state(7)
                elif uid["pricerange"] != "" and uid["pricerange"] == ls.pricerange_spellcheck(uid["pricerange"], 3):
                    self._preferred_pricerange = uid["pricerange"]
                    self.add_speech("I understood that you are interested in a {} {} restaurant in the {} area, is all this information correct?".format(self._preferred_pricerange, self._preferred_food, self._preferred_area))
                    self.set_state(8)

        elif self.get_state() == 7:  # Spelling check (pricerange)
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_pricerange = self._probable_pricerange
                self._probable_pricerange = ""
                self.add_speech("Very well, I understood that you are interested in a {} {} restaurant in the {} area, is all this information correct?".format(self._preferred_pricerange, self._preferred_food, self._preferred_area))
                self.set_state(8)
            elif dialog_act == "negate" or dialog_act == "deny":
                self._probable_pricerange = ""
                self.add_speech("Okay. Would you like to state again what price range you are interested in?")
                self.set_state(6)
            elif dialog_act == "restart":
                self._probable_pricerange = ""
                self.add_speech("Alright. Let's start over, then. What type of food are you interested in?")
                self.set_state(2)

        elif self.get_state() == 8:  # Suggest restaurant
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                # do something to get restaurant.
                if True == True:  # Found a restaurant!
                    self.add_speech("I have found a restaurant that matches your requirements human! It is the '{}' restaurant. Would you like more information, or is my function hereby fulfilled?")
                    self.set_state(9)
                elif False == False:  # Didn't find a restaurant
                    self.add_speech("I'm sorry, human. I did not find a restaurant which matches the given requirements. I will now terminate.")
                    self.set_state(11)

        elif self.get_state() == 9:  # Give information
            pass

        elif self.get_state() == 10:  # Could not find information
            pass

        elif self.get_state() == 11:  # Goodbye (terminate)
            self.add_speech("I am happy that I was able (or tried) to assist. Goodbye human.")
            self._terminated = True
    
    def classifier_handler(self, inp: str):
        return self._classifier.predict_act(inp)

    def add_speech(self, string: str):
        self._storedstring += string + "\n"

    def output_handler(self):
        """Manually defined function for handling output strings"""
        # if self.get_state() == 1:  # Hello
        #     return "Hello, human. I can help you find a restaurant based on your preferences."
        # elif self.get_state() == 2:  # Ask for food preference
        #     # do something to extract food from utterance
        #     return "What kind of cuisine are you interested in?"
        # elif self.get_state() == 3:  # Suggest spelling (food)
        #     return "I could not find anything related to {}, are you perhaps interested in {}?".format(self._preferred_food, self.statedfoodlookup)
        # elif self.get_state() == 4:  # Ask for area preference
        #     return "What area would you like to find a restaurant in?"
        # elif self.get_state() == 5:  # Suggest spelling (area)
        #     return "I could not find anything related to {}, are you perhaps interested in {}?".format(self._preferred_area, self.statedarealookup)
        # elif self.get_state() == 6:  # Ask for price preference
        #     return "What price range are you looking for?"
        # elif self.get_state() == 7:  # Suggest spelling (price)
        #     return "I could not find any price range related to {}, are you perhaps interested in a {} price range?".format(self._preferred_pricerange, self.statedpricelookup)
        # elif self.get_state() == 8:  # Suggest restaurant (db lookup)
        #     return "I would like to suggest you the restaurant '{}'."
        # elif self.get_state() == 9:  # Give information
        #     # We need a check here for the type of information we need
        #     return "The {} you're looking for is {}.".format(self.reqinfotype, self.reqinfo)
        # elif self.get_state() == 10:  # Information unavailable
        #     return "I'm sorry, I could not find the {} of the restaurant '{}'.".format(self.reqinfotype)
        # elif self.get_state() == 11:  # Goodbye (terminate)
        #     self._terminated = True
        #     return "Goodbye."
        # else:
        #     return "Error"
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
