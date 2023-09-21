from text_classification import RuleBaselineClassifier

class FiniteStateMachine:
    def __init__(self, startstate: int=0, endstate: int=18):
        self._state = startstate
        self._start = startstate
        self._end = endstate
        self._classifier = RuleBaselineClassifier()  # The output of the function must be a dialog_act as in a string!
        # self._transitionstruct = statestructure  # Manually programmed in, allows for easier definitions of logic.
        self._terminated = False
    
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
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                # do something to extract food/area/price from utterance
                self.set_state(3)
            else:
                self.set_state(2)
        elif self.get_state() == 2:  # Ask for food preference
            # do something to extract food from utterance
            pass
        elif self.get_state() == 3:  # Food preference check
            # check whether food preference was extracted properly from previous state
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