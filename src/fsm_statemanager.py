import pandas as pd

try:
    from src import classifiers
    from src import utterance_extraction_recommendation as uer
    from src import levenshtein_spellchecking as ls
except:
    import classifiers
    import utterance_extraction_recommendation as uer
    import levenshtein_spellchecking as ls

levenshtein_distance = 3

class FiniteStateMachine:
    def __init__(
        self,
        restaurant_data: pd.DataFrame,
        configuration: dict,
        classifier=None,
        startstate: int = 0,
        endstate: int = 18,
    ):
        """Initialises a Finite State Machine Dialogue Manager. Requires a dataframe containing data on
        restaurants, the configuration of the system as a whole, a classifier which predicts dialogue acts
        from given user utterances (the RuleBaselineClassifier is used by default). The startstate
        may also be modified (although heavily discouraged) and the endstate may also be modified
        (discouraged)."""
        self._state = startstate
        self._start = startstate
        self._end = endstate
        self._configuration = configuration
        self._storedstring = ""  # String to output to user.
        if classifier == None:
            self._classifier = (
                classifiers.RuleBaselineClassifier()
            )  # The output of the function must be a dialog_act as in a string!
        else:
            self._classifier = classifier  # classifier needs to be trained beforehand
        self._terminated = False
        self._restaurant_db = restaurant_data

        self._probable_food = (
            ""  # Variables to hold extracted preferences until they can be confirmed.
        )
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
        """Handles the logic loop of the finite state machine dialogue manager; user utterances are fed through the FSM network in
        combination with the current state and the classifier, and the system utterance is given back to the current process"""
        self.input_handler(inp)
        return self.output_handler()

    def input_handler(self, inp: str):
        """Manually defined function for handling logic with string inputs which represent user utterances.
        This function has been manually coded to reflect the FSM diagram, and user utterances are classified
        for dialog acts, where the system output depends on the dialog act and the information present in
        the user utterance."""
        # inp = inp.lower()
        if self.get_state() == 1:  # Hello
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "hello":
                if self._configuration["informal"]:
                    self.add_speech(
                        "Hello, I can help you find a suitable restaurant for your needs. \n"
                        "Tell me, What kind of cuisine are you interested in?"
                    )
                else:
                    self.add_speech(
                        "Hello, human. I can help you find a restaurant based on your preferences. \n"
                        "Tell me, what kind of cuisine do you fancy?"
                    )
                self.set_state(2)
                return
            if dialog_act == "inform":

                uid = uer.info_in_utterance( utterance=inp, restaurant_df=self._restaurant_db)
                # Utterance Information Dictionary

                if ls.spellcheck(inp,"dontcare"):
                    uid = uer.info_in_utterance(utterance="random", restaurant_df=self._restaurant_db)
                    self.add_speech(
                        "I interpreted that you don't care at all what kind of restaurant you'd like to visit, therefore"
                        " \neverything goes and a restaurant will be chosen at random.")
                    self._preferred_food = uid["food"]
                    self._preferred_area = uid["area"]
                    self._preferred_pricerange = uid["pricerange"]
                    self.add_speech("Do you have any further requirements?")
                    self.set_state(8)
                    return


                if uid["food"] == "":  # No valid food detected.
                    self._preferred_food = uid["food"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm sorry. I could not understand what cuisine you want. "
                            "Let me know the type of cuisine you are interested in."
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, human. I could not understand what food type you preferred. Please "
                            "state the type of cuisine you are interested in."
                        )
                    self.set_state(2)
                    return
                if uid["food"] != "" and uid["food"] != ls.spellcheck(uid["food"], "food", levenshtein_distance):
                    # Food likely misspelled
                    self.add_speech(
                        "I could not find anything related to {}, are you perhaps interested in "
                        "{}?".format(uid["food"], ls.spellcheck(uid["food"], "food", levenshtein_distance))
                    )
                    self._probable_food = ls.spellcheck(uid["food"], "food", levenshtein_distance)
                    self.set_state(3)
                    return
                else:
                    self._preferred_food = uid["food"]

                if uid["area"] == "":  # No valid area detected.
                    self._preferred_food = uid["food"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "Ok nice, {} cuisine is a great choice. I couldn't "
                            "understand what area you preferred tough. Please tell me the area you're "
                            "interested in.".format(self._preferred_food)
                        )
                    else:
                        self.add_speech(
                            "I did understand that you are interested in {} cuisine, but could "
                            "not understand what area you preferred. Please state the area you're "
                            "interested in.".format(self._preferred_food)
                        )
                    self.set_state(4)
                    return
                if uid["area"] != "" and uid["area"] != ls.spellcheck(
                    uid["area"], "area", levenshtein_distance
                ):  # Area likely misspelled
                    self.add_speech(
                        "I could not find anything related to {}, are you perhaps interested in "
                        "{}?".format(uid["area"], ls.spellcheck(uid["area"], "area", levenshtein_distance))
                    )
                    self._probable_area = ls.spellcheck(uid["area"], "area", levenshtein_distance)
                    self.set_state(5)
                    return
                else:
                    self._preferred_area = uid["area"]

                if uid["pricerange"] == "":  # No valid price range detected.
                    self._preferred_food = uid["food"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I did understand you are interested in {} cuisine in the {} area, but could "
                            "not understand what price range you wanted. Please tell me the price range "
                            "you're looking for.".format(
                                self._preferred_food, self._preferred_area
                            )
                        )
                    else:
                        self.add_speech(
                            "I did understand you are interested in {} cuisine in the {} area, but could "
                            "not understand what price range you preferred. Please state the price range "
                            "you're looking for.".format(
                                self._preferred_food, self._preferred_area
                            )
                        )
                    self.set_state(6)
                    return

                if uid["pricerange"] != "" and uid[
                    "pricerange"
                ] != ls.spellcheck(uid["pricerange"], "pricerange", levenshtein_distance):
                    # Price range likely misspelled
                    self.add_speech(
                        "I could not find any price range to {}, are you perhaps interested in a {} "
                        "price range?".format(
                            uid["pricerange"],
                            ls.spellcheck(uid["pricerange"], "pricerange", levenshtein_distance),
                        )
                    )
                    self._probable_pricerange = ls.spellcheck(
                        uid["pricerange"], "pricerange", levenshtein_distance
                    )
                    self.set_state(7)
                    return

                else:
                    self._preferred_pricerange = uid["pricerange"]
                if self._configuration["informal"]:
                    self.add_speech(
                        "Alright perfect! \nI think I got all your preferences, do you have any further requirements?"
                    )
                else:
                    self.add_speech(
                        "Alright human, I think I got all your preferences. Do you have any further requirements?"
                    )
                self.set_state(8)
                return

            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Hmm, I'm not sure I understand what you meant. Could you please try explaining it differently?"
                    )
                else:
                    self.add_speech(
                        "I did not detect an intent to supply information, human. Please try again."
                    )
                self.set_state(1)
                return

        elif self.get_state() == 2:  # Ask for food preference
            # do something to extract food from utterance
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(
                    utterance=inp, restaurant_df=self._restaurant_db
                )  # Utterance Information Dictionary

                if ls.spellcheck(inp,"dontcare"):
                    uid = uer.info_in_utterance(utterance="random", restaurant_df=self._restaurant_db)
                    self.add_speech("I interpreted that you don't care at all what kind of food you'd like, a cuisine will be chosen at random.")
                    self._preferred_food = uid["food"]
                    self.add_speech(
                        "The {} cuisine was chosen, what area are you "
                        "looking for?".format(self._preferred_food)
                    )
                    self.set_state(4)
                    return

                if uid["food"] == "":  # No valid food detected.
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I couldn't catch that. What type of cuisine are you in the mood for?"
                            " Feel free to tell me again!"
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, human. I could not understand what food type you preferred. Please "
                            "state the type of cuisine you are interested in."
                        )
                    self.set_state(2)
                    return
                elif ls.spellcheck(inp,"dontcare"):
                    self._preferred_food = uid["food"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "Ok you have stated that you have no preference at all for the cuisine. What area are you interested in?"
                        )
                    else:
                        self.add_speech(
                            "You have stated that you have no preference at all for the cuisine type of the "
                            "restaurant, human. What area are you interested in?"
                        )
                    self.set_state(4)
                    return
                elif uid["food"] != ls.spellcheck(uid["food"], "food", levenshtein_distance):
                    self.add_speech(
                        "I could not find any food related to {}, are you perhaps interested in {} "
                        "cuisine?".format(
                            uid["food"], ls.spellcheck(uid["food"], "food", levenshtein_distance)
                        )
                    )
                    self._probable_food = ls.spellcheck(uid["food"], "food", levenshtein_distance)
                    self.set_state(3)
                    return
                elif uid["food"] != "" and uid["food"] == ls.spellcheck(
                    uid["food"], "food", levenshtein_distance
                ):
                    self._preferred_food = uid["food"]
                    self.add_speech(
                        "I understood that you are interested in {} cuisine, what area are you "
                        "looking for?".format(self._preferred_food)
                    )
                    self.set_state(4)
                    return
                else:
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm so sorry! I did not understand what you meant. Can you please clarify "
                            "what kind of food you are looking for?"
                        )
                    else:
                        self.add_speech(
                            "Sorry, human. I did not understand what you meant. Can you please clarify "
                            "what kind of food you are looking for?"
                        )
                    return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Ok, please state the type of food you are interested in. "
                        "Or, if you want, we can just restart the process."
                    )
                else:
                    self.add_speech(
                        "Human, please state the type of food you are interested in. "
                        "Or, if you prefer, we can start over."
                    )
                return

        elif self.get_state() == 3:  # Food spelling check
            # check whether food preference was extracted properly from previous state
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_food = self._probable_food
                self._probable_food = ""
                if self._configuration["informal"]:
                    self.add_speech(
                        "Ok nice, you are interested in {} cuisine. What area are you "
                        "looking for?".format(self._preferred_food)
                    )
                else:
                    self.add_speech(
                        "Very well, you are interested in {} cuisine. What area are you "
                        "looking for?".format(self._preferred_food)
                    )
                self.set_state(4)
                return
            elif dialog_act == "negate" or dialog_act == "deny":
                if self._configuration["informal"]:
                    self.add_speech(
                        "Okay. Could you tell me again what cuisine you are interested in?"
                    )
                else:
                    self.add_speech(
                        "Okay. Would you like to state again what cuisine you are interested in?"
                    )
                self.set_state(2)
                return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Is my recommendation good? Or would you like to start over?"
                    )
                else:
                    self.add_speech(
                        "Is my recommendation correct human? Or would you perhaps like to start over?"
                    )
                return

        elif self.get_state() == 4:  # Ask for area preference
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(
                    utterance=inp, restaurant_df=self._restaurant_db
                )  # Utterance Information Dictionary

                if ls.spellcheck(inp,"dontcare"):
                    uid = uer.info_in_utterance(utterance="random", restaurant_df=self._restaurant_db)
                    self.add_speech("I interpreted that you don't care at all what area to go, an area will be chosen at random.")
                    self._preferred_area = uid["area"]
                    self.add_speech(
                        "The {} area was chosen, What price range are you interested in?".format(self._preferred_area)
                        )
                    self.set_state(6)
                    return

                if uid["area"] == "":  # No valid area detected.
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm sorry. I could not understand what area you preferred. Please state"
                            " the area you'd like your restaurant to be in."
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, human. I could not understand what area you preferred. Please state"
                            " the area in which you want to find a restaurant."
                        )
                    self.set_state(4)
                    return
                if ls.spellcheck(inp,"dontcare"):  # No valid area detected.
                    self._preferred_area = uid["area"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "You have stated that you have no preference at all for the restaurant's area. "
                            "But what price range are you interested in?"
                        )
                    else:
                        self.add_speech(
                            "You have stated that you have no preference at all for the restaurant's area, "
                            "human. What price range are you interested in?"
                        )
                    self.set_state(6)
                    return
                elif uid["area"] != ls.spellcheck(uid["area"], "area", levenshtein_distance):
                    self.add_speech(
                        "I could not find any area related to {}, are you perhaps interested in the {} "
                        "area?".format(uid["area"], ls.spellcheck(uid["area"], "area", levenshtein_distance))
                    )
                    self._probable_area = ls.spellcheck(uid["area"], "area", levenshtein_distance)
                    self.set_state(5)
                    return
                elif uid["area"] != "" and uid["area"] == ls.spellcheck(
                    uid["area"], "area", levenshtein_distance
                ):
                    self._preferred_area = uid["area"]
                    self.add_speech(
                        "I understood that you are interested in restaurants in the {} area, what price"
                        " range are you looking for?".format(uid["area"])
                    )
                    self.set_state(6)
                    return
                else:
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm Sorry. I did not understand what you meant. Could you please clarify what "
                            "kind of area you are interested in?"
                        )
                    else:
                        self.add_speech(
                            "Sorry, human. I did not understand what you meant. Would you clarify what kind"
                            " of area you are interested in?"
                        )
                    return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Ok then, please state the area you are interested in! "
                        "Or, if you want, we can just start over."
                    )
                else:
                    self.add_speech(
                        "Human, please state the area you are interested in. "
                        "Or, if you prefer, we can start over."
                    )
                return

        elif self.get_state() == 5:  # Suggest spelling (area)
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_area = self._probable_area
                self._probable_area = ""
                self.add_speech(
                    "Very well, you are interested in a restaurant in the {} area. "
                    "What price range are you interested in?".format(
                        self._preferred_area
                    )
                )
                self.set_state(6)
                return
            elif dialog_act == "negate" or dialog_act == "deny":
                self.add_speech(
                    "Okay. Would you like to state again what area you are interested in?"
                )
                self.set_state(4)
                return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Is my recommendation good? "
                        "Or would you just like to start over?"
                    )
                else:
                    self.add_speech(
                        "Is my recommendation correct human? "
                        "Or would you perhaps like to start over completely?"
                    )
                return

        elif self.get_state() == 6:  # Ask for pricerange preference
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                uid = uer.info_in_utterance(
                    utterance=inp, restaurant_df=self._restaurant_db
                )  # Utterance Information Dictionary

                if ls.spellcheck(inp,"dontcare"):
                    uid = uer.info_in_utterance(utterance="random", restaurant_df=self._restaurant_db)
                    self.add_speech("I interpreted that you don't care at all about a price range, it will be chosen at random.")
                    self._preferred_pricerange = uid["price_range"]
                    self.add_speech(
                        "The {} price range was chosen, do you have any further requirements?".format(self._preferred_pricerange)
                        )
                    self.set_state(8)
                    return

                if uid["pricerange"] == "":  # No valid area detected.
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm sorry but I couldn't understand what price range you preferred. "
                            "Please rephrase and tell me again the price range you're interested in."
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, human. I could not understand what price range you preferred. "
                            "Please state the price range of restaurants you are interested in."
                        )
                    self.set_state(6)
                    return
                if ls.spellcheck(inp,"dontcare"):
                    self._preferred_pricerange = uid["pricerange"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "You've said that you have no preference at all for the price "
                            "of the restaurant. Do you have any further requirements?"
                        )
                    else:
                        self.add_speech(
                            "You have stated that you have no preference at all for the price "
                            "of the restaurant, human. Do you have any further requirements?"
                        )
                    self.set_state(8)
                    return
                elif uid["pricerange"] != ls.spellcheck(
                    uid["pricerange"], "pricerange", levenshtein_distance
                ):
                    self.add_speech(
                        f"I could not find any {uid['pricerange']} price range, are you perhaps interested in a {ls.spellcheck(uid['pricerange'], 'pricerange', levenshtein_distance)} "
                        "restaurant?"
                    )
                    self._probable_pricerange = ls.spellcheck(
uid["pricerange"], "pricerange", levenshtein_distance)
                    self.set_state(7)
                    return
                elif uid["pricerange"] != "" and uid[
                    "pricerange"
                ] == ls.spellcheck(uid["pricerange"], "pricerange", levenshtein_distance):
                    self._preferred_pricerange = uid["pricerange"]
                    self.add_speech(
                        f"I understood that you are interested in a {self._preferred_pricerange} restaurant. Do you have any "
                        "further requirements?"
                    )
                    self.set_state(8)
                    return
                else:
                    if self._configuration["informal"]:
                        self.add_speech(
                            "Sorry, I did not understand what you meant. Could you clarify "
                            "what kind of price range you are looking for?"
                        )
                    else:
                        self.add_speech(
                            "Sorry, human. I did not understand what you meant. Would you clarify "
                            "what kind of price range you are looking for?"
                        )
                    return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Please tell me the price range you are looking for. "
                        "Or, if you want, we can start over."
                    )
                else:
                    self.add_speech(
                        "Human, please state the price range you are looking for. "
                        "Or, if you want, we can start over."
                    )
                return

        elif self.get_state() == 7:  # Spelling check (pricerange)
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                self._preferred_pricerange = self._probable_pricerange
                self._probable_pricerange = ""
                self.add_speech(
                    f"Alright, I understood that you are interested in a {self._preferred_pricerange} restaurant. "
                    "Do you have any further requirements?"
                )
                self.set_state(8)
                return
            elif dialog_act == "negate" or dialog_act == "deny":
                self._probable_pricerange = ""
                self.add_speech(
                    "Okay. Could you rephrase what price range you are interested in?"
                )
                self.set_state(6)
                return
            elif dialog_act == "restart":
                self._probable_pricerange = ""
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Are my recommendation good? Or would you like to start over?"
                    )
                else:
                    self.add_speech(
                        "Are my recommendation correct human? Or would you perhaps like to start over?"
                    )
                return

        elif self.get_state() == 8:  # Additional requests
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "inform":
                # do something to extract additional information here
                uid = uer.info_in_utterance(
                    utterance=inp, restaurant_df=self._restaurant_db
                )
                if uid["preference"] == "":
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm sorry but I couldn't understand your additional requirement. "
                            "Please rephrase and tell me again."
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, I could not understand your additional requirement. "
                            "Please state it again."
                        )
                    self.set_state(8)
                    return
                if ls.spellcheck(inp,"dontcare"):
                    self._additional_requirements = uid["preference"]
                    if self._configuration["informal"]:
                        self.add_speech(
                            "You've said that you have no preference for the additional requirement. "
                            "Do you have any further requirements?"
                        )
                    else:
                        self.add_speech(
                            "You have stated that you have no preference for the additional requirement. "
                            "Do you have any further requirements?"
                        )
                    self.set_state(8)
                    return
                elif uid["preference"] != ls.spellcheck(
                    uid["preference"], "preference", levenshtein_distance
                ):
                    self.add_speech(
                        f"I could not find any suitable restaurant for the additional requirement '{uid['preference']}'. "
                        "Are you perhaps interested in something else?"
                    )
                    self.set_state(8)
                    return
                elif uid["preference"] != "" and uid[
                    "preference"
                ] == ls.spellcheck(uid["preference"], "preference", levenshtein_distance):
                    self._additional_requirements = uid["preference"]
                    self.add_speech(
                        f"I understood that you have an additional requirement for '{self._additional_requirements}'. "
                        "Do you have any further requirements?"
                    )
                    self.set_state(8)
                    return
                else:
                    if self._configuration["informal"]:
                        self.add_speech(
                            "Sorry, I did not understand what you meant by your additional requirement. "
                            "Could you clarify it?"
                        )
                    else:
                        self.add_speech(
                            "Sorry, I did not understand your additional requirement. Could you clarify it?"
                        )
                    return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            elif dialog_act in ["negate", "deny"]:
                # This is going to be the only place where we mention *everything* now.
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
                self.add_speech(
                    "All right. I understand that you are interested in a {}, {} restaurant in {} area "
                    "with the following requirements: {}. "
                    "Is that correct?".format(
                        self._represent_string_pricerange,
                        self._represent_string_food,
                        self._represent_string_area,
                        self._additional_requirements,
                    )
                )
                self.set_state(9)
                # return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "All right, do you have any additional requirements for the restaurant? "
                        "Or would you like to start over?"
                    )
                else:
                    self.add_speech(
                        "Human, do you have any (additional) requirements for the restaurant? "
                        "Or would you like to start over?"
                    )

        elif self.get_state() == 9:  # Suggest restaurant
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "ack" or dialog_act == "affirm" or dialog_act == "confirm":
                # do something to get restaurant.
                self._possible_recommendations = uer.provide_recommendations(self._restaurant_db, self._preferred_food, self._preferred_pricerange, self._preferred_area)  # type: ignore
                if self._possible_recommendations.shape[0] >= 1:  # Found a restaurant!
                    (
                        self._probable_restaurant,
                        self._possible_recommendations,
                    ) = uer.pop_recommendation(self._possible_recommendations)

                    #Fetching the reasoning behind the additional requirement
                    rec_rests, preference_reason = uer.preference_reasoning(self._possible_recommendations, self._additional_requirements) # type: ignore

                    if self._configuration["informal"]:
                        self.add_speech(
                            f"Ok great, I have found a restaurant that matches your requirements! "
                            f"It is the '{self._probable_restaurant}' restaurant. "
                            f"{preference_reason} Would you like more information?"
                        )
                    else:
                        self.add_speech(
                            f"I have found a restaurant that matches your exact requirements human! "
                            f"It is the '{self._probable_restaurant}' restaurant. "
                            f"{preference_reason} Would you like additional information?"
                        )
                    self.set_state(10)
                    return

                else:  # Didn't find a restaurant
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm sorry. I did not find a restaurant which matches your requirements."
                            " Would you like to start over or quit the program?"
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, human. I did not find a restaurant which matches the given requirements."
                            " Would you like to start over or is my function fulfilled?"
                        )
                    self.set_state(11)
                    return
            elif dialog_act in ["negate", "deny"]:
                self.add_speech(
                    "Very well, let us start over, then. What kind of cuisine are you looking for?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Am I correct that you're looking for a {}, {} restaurant in {} area "
                        "with the following requirements: {}?".format(
                            self._represent_string_pricerange,
                            self._represent_string_food,
                            self._represent_string_area,
                            self._additional_requirements,
                        )
                    )
                else:
                    self.add_speech(
                        "Human, are you correctly looking for a {}, {} restaurant in {} area "
                        "with the following requirements: {}?".format(
                            self._represent_string_pricerange,
                            self._represent_string_food,
                            self._represent_string_area,
                            self._additional_requirements,
                        )
                    )
                return

        elif self.get_state() == 10:  # Give information
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "reqmore":
                info_dict = uer.get_restaurant_info(
                    restaurants_df=self._restaurant_db,
                    restaurantname=self._probable_restaurant,
                )  # type: ignore
                self.add_speech("Here's some information:")
                self.add_speech(f"Address: {info_dict['addr']}")
                self.add_speech(f"Phone number: {info_dict['phone']}")
                self.add_speech(f"Zipcode: {info_dict['postcode']}")
                if self._configuration["informal"]:
                    self.add_speech(
                        "Is there anything else I can help you with? Or is our conversation over?"
                    )
                else:
                    self.add_speech(
                        "Is there anything else I can be of assistance with? "
                        "Can I perhaps provide the same information again, or is our conversation finished?"
                    )
                return
            elif dialog_act == "reqalts":
                if self._configuration["informal"]:
                    self.add_speech(
                        "Not interested in this restaurant? Okay, let me see if I can find something else."
                    )
                else:
                    self.add_speech(
                        "Not interested in this restaurant? Okay human, let me see if I can find an alternative."
                    )
                if len(self._possible_recommendations) < 1:
                    if self._configuration["informal"]:
                        self.add_speech(
                            "I'm sorry I did not find a restaurant which matches your "
                            "requirements. Would you like to start over or quit?"
                        )
                    else:
                        self.add_speech(
                            "I'm sorry, human. I did not find a restaurant which matches the given "
                            "requirements. Would you like to start over or is my function fullfilled?"
                        )
                    self.set_state(11)
                    return
                (
                    self._probable_restaurant,
                    self._possible_recommendations,
                ) = uer.pop_recommendation(self._possible_recommendations)
                if self._configuration["informal"]:
                    self.add_speech(
                        "I have found another restaurant that matches your requirements! "
                        "It is the '{}' restaurant. Would you like more information about this "
                        "restaurant?".format(self._probable_restaurant)
                    )
                else:
                    self.add_speech(
                        "I have found another restaurant that matches your requirements human! "
                        "It is the '{}' restaurant. Would you like more information about this "
                        "restaurant? Or am I done?".format(self._probable_restaurant)
                    )
                self.set_state(10)
                return
            elif dialog_act == "deny" or dialog_act == "bye":
                if self._configuration["informal"]:
                    self.add_speech("Very well, would you like to start over or is my function fullfilled?")
                else:
                    self.add_speech("Alright, human. Would you like to start over or is my function fullfilled?")
                self.set_state(11)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        f"Would you like more information about the {self._probable_restaurant} restaurant, "
                        "or maybe you want alternative restaurants?"
                    )
                else:
                    self.add_speech(
                        f"Human, would you like more information about the {self._probable_restaurant} restaurant, "
                        "or perhaps you want alternative restaurants?"
                    )
                return

        elif self.get_state() == 11:
            dialog_act = self.classifier_handler(inp)
            if dialog_act == "bye" or dialog_act == "thankyou":
                if self._configuration["informal"]:
                    self.add_speech("Alright, I will now close. I hope I was useful!")
                else:
                    self.add_speech(
                        "Alright human, I will now terminate. Did I perform amiably?"
                    )
                self.set_state(12)
                return
            elif dialog_act == "restart":
                self.add_speech(
                    "Alright. Let's start over, then. What type of food are you interested in?"
                )
                self.set_state(2)
                return
            else:
                if self._configuration["informal"]:
                    self.add_speech(
                        "Are we finished or would you want to take another go?"
                    )
                else:
                    self.add_speech(
                        "Human, is my function hereby fulfilled, or do you want to start over?"
                    )
                return

        elif self.get_state() == 12:  # Goodbye (terminate)
            if self._configuration["informal"]:
                self.add_speech("Good to hear I was able to assist. Goodbye.")
            else:
                self.add_speech(
                    "I am happy that I was able (or tried) to assist. Goodbye human."
                )
            self._terminated = True

    def classifier_handler(self, inp: str):
        """Feeds user inputs through the dialog act classifier model."""
        return self._classifier.predict_act(inp)

    def add_speech(self, string: str):
        """Handles string inputs. Uses given configuration to determine
        whether output should be in caps or not, and formats it to use multiple
        lines of input."""
        if self._configuration["capitals"]:
            string = string.upper()
        self._storedstring += string + "\n"

    def output_handler(self):
        """Returns the stored string (added through self.add_speech()) while
        emptying the stored string afterwards."""
        tmp = self._storedstring
        self._storedstring = ""
        return tmp

    def get_state(self):
        """Returns the current state the FSM is currently evaluating."""
        return self._state

    def set_state(self, state: int):
        """Sets a new state for the FSM to evaluate. Also flags that the
        FSM has reached the terminal state if this is the case."""
        self._state = state
        if self._state == self._end:
            self._terminated = True

    def reset(self):
        """Resets the FSM, restoring the state to the start state
        and clearing only the terminated flag."""
        self._state = self._start
        self._terminated = False
