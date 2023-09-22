import csv
import pandas as pd
from Levenshtein import distance

# Reading the csv
csv_file = 'Data/restaurant_info.csv'
df = pd.read_csv(csv_file, sep=',')
print()

# Spellchecking for food
def food_spellcheck(food, threshold=2):
    # Get unique values from the column and convert them to a list
    unique_foods_list = df['food'].unique().tolist()

    closest_food = None # No suggestion yet
    min_distance = threshold # How many steps to edit to the correct word (delete, change, add letters etc.)

    # Finding the closest word that the system knows
    for known_food in unique_foods_list:
        d = distance(food, known_food)
        if d < min_distance:
            min_distance = d
            closest_food = known_food

    return closest_food

# Spellchecking for area
def area_spellcheck(area, threshold=2):
    known_areas = ["north", "west", "east", "south", "centre"]
    closest_area = None
    min_distance = threshold

    for known_area in known_areas:
        d = distance(area, known_area)
        if d < min_distance:
            min_distance = d
            closest_area = known_area

    return closest_area

# Spellchecking for pricerange
def pricerange_spellcheck(pricerange, threshold=2):
    closest_pricerange = None
    min_distance = threshold
    known_priceranges = ["budget-friendly", "cheap", "affordable", "economical",
                         "low-cost", "thrifty", "wallet-friendly", "average", "moderate", "mid-range", "reasonable",
                         "fair",
                         "standard", "middle-of-the-road", "not too pricey", "decently priced", "luxury", "high-end",
                         "expensive", "upscale", "premium",
                         "lavish", "top-tier", "pricey", "exclusive"]

    for known_pricerange in known_priceranges:
        d = distance(pricerange, known_pricerange)
        if d < min_distance:
            min_distance = d
            closest_pricerange = known_pricerange

    return closest_pricerange

def test_spellchecking():
    # Input word to be corrected
    input_word = input("Enter a word: ").lower()

    # Get the suggested correction
    correction_food = food_spellcheck(input_word)

    if correction_food:
        print(f"Suggested pricerange correction: {correction_food}")
    else:
        print("No suggestion found for food.")

    correction_pricerange = pricerange_spellcheck(input_word)

    if correction_pricerange:
        print(f"Suggested pricerange correction: {correction_pricerange}")
    else:
        print("No suggestion found for pricerange.")


    correction_area = area_spellcheck(input_word)
    if correction_area:
        print(f"Suggested area correction: {correction_area}")
    else:
        print("No suggestion found for area.")



