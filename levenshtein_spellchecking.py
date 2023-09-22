import csv
import pandas as pd
from Levenshtein import distance

# Reading the csv
csv_file = '/Users/berkeyazan/Desktop/restaurant_info.csv'
df = pd.read_csv(csv_file, sep=';')

# Spellchecking for food
def food_spellceck(food, threshold=2):
    # Extract values from the 4th column
    foods_in_column_4 = df.iloc[:, 3]
    # Get unique values from the column and convert them to a list
    unique_foods_list = foods_in_column_4.unique().tolist()

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
def area_spellceck(area, threshold=2):
    known_areas = ["north", "west", "east", "south"]
    closest_area = None
    min_distance = threshold

    for known_area in known_areas:
        d = distance(area, known_area)
        if d < min_distance:
            min_distance = d
            closest_area = known_area

    return closest_area

# Spellchecking for pricerange
def pricerange_spellceck(pricerange, threshold=2):
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


# Input word to be corrected
input_word = input("Enter a word: ").lower()

# Get the suggested correction
correction_food = food_spellceck(input_word)

if correction_food:
    print(f"Suggested pricerange correction: {correction_food}")
else:
    print("No suggestion found for food.")

correction_pricerange = pricerange_spellceck(input_word)

if correction_pricerange:
    print(f"Suggested pricerange correction: {correction_pricerange}")
else:
    print("No suggestion found for pricerange.")


correction_area = area_spellceck(input_word)
if correction_area:
    print(f"Suggested area correction: {correction_area}")
else:
    print("No suggestion found for area.")



