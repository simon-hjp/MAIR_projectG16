import text_classification
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
            self._classifier = text_classification.RuleBaselineClassifier()  # The output of the function must be a dialog_act as in a string!
        else:
            self._classifier = classifier # classifier needs to be trained beforehand
        self._terminated = False
        self._restaurant_db = restaurant_data
        self._preferred_food = ""
        self._preferred_area = ""
        self._preferred_pricerange = ""
        # self._transitionstruct = statestructure  # Manually programmed in, allows for easier definitions of logic.
    
    # def state_transition(self, dialog_act: str):
        # if dialog_act in self._transitionstruct[self._state].keys() and self._terminated is False:
        #     tmp = self._transitionstruct[self._state][dialog_act]
        #     self._state = tmp
        #     if self._end is not None:
        #         if self._end == self._state:
        #             self._terminated = True
        #     return self._state  # Return state so state can be tracked
        # else:
        #     return -1  # State -1 should be reserved to visualise illegal operations
        # if self.get_state() == 1:
    
    def logic(self, inp: str):
        # One small problem with the current structure; currently the hello state system utterance will always be skipped, not sure if that's intentional.
        self.input_handler(inp)
        return self.output_handler()
    
    def input_handler(self, inp: str):
        """Manually defined function for handling logic with string inputs"""
        if self.get_state() == 1:  # Hello
            self.output_handler()
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "hi":
                self.add_speech("Hello, human. I can help you find a restaurant based on your preferences. What kind of cuisine are you interested in?")
                self.set_state(2)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(utterance=inp, df=self._restaurant_db)  # Utterance Information Dictionary

                if uid["food"] == "":
                    self.add_speech("I'm sorry, human. I could not understand what food type you preferred. Try again.")
                    self.set_state(1)
                elif uid["food"] != ls.food_spellcheck(uid["food"], 3):
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["food"], ls.food_spellcheck(uid["food"], 3)))
                    self.set_state(3)

                if uid["area"] == "":
                    self.add_speech("I'm sorry, human. I could not understand what area you preferred. Try again.")
                    self.set_state(1)
                elif uid["area"] != ls.area_spellcheck(uid["area"], 3):
                    self.add_speech("I could not find anything related to {}, are you perhaps interested in {}?".format(uid["area"], ls.food_spellcheck(uid["area"], 3)))
                    self.set_state(3)

                if uid["pricerange"] == "":
                    self.add_speech("I'm sorry, human. I could not understand what price range you preferred. Try again.")
                    self.set_state(1)
                elif uid["pricerange"] != ls.area_spellcheck(uid["pricerange"], 3):
                    self.add_speech("I could not find any price range to {}, are you perhaps interested in a {} price range?".format(uid["pricerange"], ls.food_spellcheck(uid["pricerange"], 3)))
                    self.set_state(3)
            else:
                self.set_state(1)
        elif self.get_state() == 2:  # Ask for food preference
            # do something to extract food from utterance
            pass
        elif self.get_state() == 3:  # Food spelling check
            # check whether food preference was extracted properly from previous state
            pass
        elif self.get_state() == 4:  # Ask for area preference
            pass
        elif self.get_state() == 5:  # Suggest spelling (area)
            pass
        elif self.get_state() == 6:  # Ask for pricerange preference
            pass
        elif self.get_state() == 7:  # Spelling check(pricerange)
            pass
        elif self.get_state() == 8:  # Suggest restaurant
            pass
        elif self.get_state() == 9:  # Give information
            pass
        elif self.get_state() == 10:  # Could not find information
            pass
        elif self.get_state() == 11:  # Goodbye (terminate)
            print("Goodbye.")
            self._terminated = True
            pass
        # return self.transition_label(self._classifier.predict_act(inp))
    
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
