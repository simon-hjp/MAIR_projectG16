import pandas as pd
# from levenshtein_spellchecking import food_spellcheck, area_spellcheck, pricerange_spellcheck

def info_in_utterance(utterance: str, df: pd.DataFrame):
    # Initialize variables
    area = ""
    food = ""
    pricerange = ""

    # Looking for food
    # Extract values from the 4th column
    foods_in_column_4 = df.iloc[:, 3]

    # Get unique values from the column and convert them to a list
    unique_foods_list = foods_in_column_4.unique().tolist()

    # Search for the input in the unique_values_list
    for food in unique_foods_list:
        if food in utterance:
            food = food
            print(f"food = {food}")
            break    

    # Check if the utterance contains words related to area (north, west, east, south)
    area_words = ["north", "west", "east", "south", "center"]
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
            pricerange = 'cheap'
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
    
    return {'food': food, 'area': area, 'pricerange': pricerange}

def provide_recommendations(restaurants_df: pd.DataFrame, req_food="", req_pricerange="", req_area="") -> str:
    """Return a restaurant recommendation based on the requested attributes by the user.
    """
    possible_recs = restaurants_df.copy()
    if req_food != "":
        possible_recs = possible_recs[possible_recs["food"]==req_food]
    if req_pricerange != "":
        possible_recs = possible_recs[possible_recs["pricerange"]==req_pricerange]
    if req_area != "":
        possible_recs = possible_recs[possible_recs["area"]==req_area]
    
    if len(possible_recs) < 1:
        return "No restaurant"
    return possible_recs["restaurantname"].sample(n=1, random_state=5).iloc[0]

def get_restaurant_info(restaurants_df: pd.DataFrame, restaurantname: str):
    """Given a restaurant name, return its information as a dictionary.

    Returns: dictionary with keys 'restaurantname', 'area', 'pricerange', 'food', 'phone', 'addr', 'postcode'.
    """
    if restaurantname not in restaurants_df["restaurantname"].to_list():
        return "Not found"
    return restaurants_df[restaurants_df["restaurantname"]==restaurantname].to_dict(orient='records')[0]

restaurant_data = pd.read_csv('Data/restaurant_info.csv')

while True:
    utterance = input("Hello, how can I help you?\nYour utterance ('1' to quit): ").lower()
    if utterance == '1':
        break
    
    info_dict = info_in_utterance(utterance, restaurant_data)
    recommendation = provide_recommendations(restaurants_df=restaurant_data, req_food=info_dict["food"], req_area=info_dict["area"], req_pricerange=info_dict["pricerange"])
    print("Recommended restaurant:", recommendation)
    print("Recommendation info:", get_restaurant_info(restaurant_data, recommendation))
