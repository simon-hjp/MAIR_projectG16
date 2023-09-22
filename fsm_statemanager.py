import text_classification
import utterance_extraction_recommendation
import pandas as pd

class FiniteStateMachine:
    def __init__(self, restaurant_data: pd.DataFrame, classifier=None, startstate: int=0, endstate: int=18):
        self._state = startstate
        self._start = startstate
        self._end = endstate
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
                self.set_state(1)
            if dialog_act == "inform":
                # do something to extract food/area/price from utterance
                # move to different state based on what is specified.
                utterance_extraction_recommendation.info_in_utterance(utterance=inp, df=self._restaurant_db)
                self.set_state(1)
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

    def output_handler(self):
        """Manually defined function for handling output strings"""
        if self.get_state() == 1:  # Hello
            return "Hello"
        elif self.get_state() == 2:  # Ask for food preference
            return ""
        elif self.get_state() == 3:  # Food preference check
            pass
        elif self.get_state() == 4:  # Spelling check (food)
            pass
        elif self.get_state() == 5:  # Suggest spelling (food)  
            pass
        elif self.get_state() == 6:  # Ask for area preference
            pass
        elif self.get_state() == 7:  # Area preference check
            pass
        elif self.get_state() == 8:  # Spelling check (area)
            pass
        elif self.get_state() == 9:  # Suggest spelling (area)
            pass
        elif self.get_state() == 10:  # Ask for price preference
            pass
        elif self.get_state() == 11:  # Price preference check
            pass
        elif self.get_state() == 12:  # Spelling check (price)
            pass
        elif self.get_state() == 13:  # Suggest spelling (price)
            pass
        elif self.get_state() == 14:  # Suggest restaurant (db lookup)
            pass
        elif self.get_state() == 15:  # Information check (known or not)
            pass
        elif self.get_state() == 16:  # Give information
            pass
        elif self.get_state() == 17:  # Information unavailable
            pass
        elif self.get_state() == 18:  # Goodbye (terminate)
            return "Goodbye"
        else:
            return "Error"

    def get_state(self):
        return self._state
    
    def set_state(self, state: int):
        self._state = state
        if self._state == self._end:
            self._terminated = True
    
    def reset(self):
        self._state = self._start
        self._terminated = False
