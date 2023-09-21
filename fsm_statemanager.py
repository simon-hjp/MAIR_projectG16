class FiniteStateMachine:
    def __init__(self, statestructure: dict, startstate: int=0, endstate: int = None):
        self._state = startstate
        self._start = startstate
        self._end = endstate
        self._transitionstruct = statestructure
        self._terminated = False
    
    def transition_label(self, action: str):
        if action in self._transitionstruct[self._state].keys() and self._terminated is False:
            tmp = self._transitionstruct[self._state][action]
            self._state = tmp
            if self._end is not None:
                if self._end == self._state:
                    self._terminated = True
            return self._state  # Return state so state can be tracked
        else:
            return -1  # State -1 should be reserved to visualise illegal operations
        
    def get_state(self):
        return self._state
    
    def reset(self):
        self._state = self._start
        self._terminated = False