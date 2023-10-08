import pandas as pd
from Levenshtein import distance

# Read the csv
try:
    csv_file = '../Data/restaurant_info.csv'
    df = pd.read_csv(csv_file, sep=',')
except:
    csv_file = 'Data/restaurant_info.csv'
    df = pd.read_csv(csv_file, sep=',')

# create word lists
foods_list = df['food'].unique().tolist()

areas_list = ["north", "west", "east", "south", "centre"]

priceranges_list = ["budget-friendly", "cheap", "affordable", "economical", "low-cost", "thrifty",
                    "wallet-friendly", "average", "moderate", "mid-range", "reasonable", "fair",
                    "standard", "middle-of-the-road", "not too pricey", "decently priced", "luxury", "high-end",
                    "expensive", "upscale", "premium", "lavish", "top-tier", "pricey", "exclusive"]

preferences_list = ['touristic', 'not touristic', 'romantic', 'not romantic',
                    'children', 'no children', 'not children', 'assigned seats',
                    'no assigned seats', 'not assigned seats']

dontcare_list = ["I don't care","dontcare","dont care","Don't care","any will do","any","whatever","whatever works",
                 "any works","idunno","no preference","I have no preference","up for anything","i dont know",
                 "don't know","i don't know","what ever","any","I do not care","care"]

def spellcheck(word: str, keyword: str, threshold: int=3):
    """Perform spellchecking of important words in the user utterance with Levenshtein edit distance."""
    if keyword == "food":
        word_list = foods_list
    elif keyword == "area":
        word_list = areas_list
    elif keyword == "pricerange":
        word_list = priceranges_list
    elif keyword == "preference":
        word_list = preferences_list
    elif keyword == "dontcare":
        word_list = dontcare_list
    else:
        raise ValueError(f'Invalid keyword for spellchecking : {keyword}.')


    closest_word = None # No suggestion yet
    min_distance = threshold # How many steps to edit to the correct word (delete, change, add letters etc.)

    # Finding the closest word that the system knows
    for known_word in word_list:
        dist = distance(word, known_word)
        if dist < min_distance:
            min_distance = dist
            closest_word = known_word

    return closest_word

def test_spellchecking():
    """Spell check a word provided by user."""
    # Input word to be corrected
    input_word = input("Enter a word: ").lower()

    # Get the suggested correction
    correction_food = spellcheck(input_word, "food")

    if correction_food:
        print(f"Suggested pricerange correction: {correction_food}")
    else:
        print("No suggestion found for food.")

    correction_pricerange = spellcheck(input_word, "pricerange")

    if correction_pricerange:
        print(f"Suggested pricerange correction: {correction_pricerange}")
    else:
        print("No suggestion found for pricerange.")


    correction_area = spellcheck(input_word, "area")
    if correction_area:
        print(f"Suggested area correction: {correction_area}")
    else:
        print("No suggestion found for area.")

if __name__ == "__main__":
    test_spellchecking()

