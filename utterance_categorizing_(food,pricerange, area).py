import csv
import pandas as pd

def info_in_utterance(utterance, csv_file):
    # Initialize variables
    area = ""
    food = ""
    pricerange = ""

    # Looking for food
    # Choose the separation symbol
    df = pd.read_csv(csv_file, sep=';')

    # Extract values from the 4th column
    foods_in_column_4 = df.iloc[:, 3]

    # Get unique values from the column and convert them to a list
    unique_foods_list = foods_in_column_4.unique().tolist()

    # Search for the input in the unique_values_list
    for food in unique_foods_list:
        if food in utterance:
            print(f"food = {food}")
            break

    # Check if the utterance contains words related to area (north, west, east, south)

    area_words = ["north", "west", "east", "south"]
    for word in area_words:
        if word in utterance:
            area = word
            print(f'area: {area}')
            break

    # Words that represent different price ranges
    cheap = ["budget-friendly", "cheap", "affordable", "economical",
             "low-cost", "thrifty", "wallet-friendly"]

    moderate = ["average", "moderate", "mid-range", "reasonable", "fair",
                "standard", "middle-of-the-road", "not too pricey", "decently priced"]

    expensive = ["luxury", "high-end", "expensive", "upscale", "premium",
                 "lavish", "top-tier", "pricey", "exclusive"]

    # Check if the utterance contains words related to pricerange (cheap, moderate, expensive)
    for cheap_words in cheap:
        if cheap_words in utterance:
            pricerange = cheap
            print(f'pricerange: {pricerange}')
            break
    for moderate_words in moderate:
        if moderate_words in utterance:
            pricerange = 'moderate'
            print(f'pricerange: {pricerange}')
            break
    for expensive_words in expensive:
        if expensive_words in utterance:
            pricerange = 'expensive'
            print(f'pricerange: {pricerange}')
            break

while True:
    csv_file = '/Users/berkeyazan/Desktop/restaurant_info.csv'
    utterance = input("Hello, how can I help you?\nYour utterance ('1' to quit): ").lower()
    if utterance == '1':
        break

    info_in_utterance(utterance, csv_file)
