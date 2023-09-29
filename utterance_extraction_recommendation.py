import pandas as pd
import numpy as np

def info_in_utterance(utterance: str, df: pd.DataFrame):
    # Initialize variables
    area = ""
    food = ""
    pricerange = ""
    preference = ""

    # Looking for food
    # Get unique values from the column and convert them to a list
    unique_foods_list = df['food'].unique().tolist()

    # Search for the input in the unique_values_list
    for foodw in unique_foods_list:
        if foodw in utterance:
            food = foodw
            print(f"food = {food}")
            break    

    # Check if the utterance contains words related to area (north, west, east, south)
    area_words = ["north", "west", "east", "south", "centre"]
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
    
    preference_words = ['touristic', 'not touristic', 'romantic', 'not romantic',
                        'children', 'no children', 'not children', 'assigned seats',
                        'no assigned seats', 'not assigned seats']
    for additional_preference in preference_words:
        if additional_preference in utterance:
            preference = additional_preference
            print(f'Additional preference: {preference}')
    
    return {'food': food, 'area': area, 'pricerange': pricerange, 'preference': preference}

def provide_recommendations(restaurants_df: pd.DataFrame, req_food="", req_pricerange="", req_area="") -> pd.DataFrame:
    """Return a restaurant recommendation based on the requested attributes by the user. If a preference 
    """
    possible_recs = restaurants_df.copy()
    if req_food != "":
        possible_recs = possible_recs[possible_recs["food"]==req_food]
    if req_pricerange != "":
        possible_recs = possible_recs[possible_recs["pricerange"]==req_pricerange]
    if req_area != "":
        possible_recs = possible_recs[possible_recs["area"]==req_area]
    # print(possible_recs)
    return possible_recs

def preference_reasoning(rec_rests: pd.DataFrame, req_consequent: str) -> tuple[pd.DataFrame, str]:
    """Reason about restaurants that could be recommended whether they satisfy the 
    user's additional preference."""
    if req_consequent == 'touristic':
        rec_rests = rec_rests[(rec_rests['pricerange']=='cheap') & (rec_rests['food_quality']=='good food')]
        return rec_rests, 'This restaurant is touristic because it offers good food for cheap prices.'
    if req_consequent == 'not touristic':
        rec_rests = rec_rests[rec_rests['food']=='romanian']
        return rec_rests, 'This restaurant is usually not considered touristic because it offers food from the unfamiliar Romanian cuisine.'
    if req_consequent == 'assigned seats':
        rec_rests = rec_rests[rec_rests['crowdedness'] == 'busy']
        return rec_rests, 'This restaurant is busy, so waiters assign the seats for you.'
    if req_consequent == 'not assigned seats' or req_consequent == 'no assigned seats':
        rec_rests = rec_rests[rec_rests['crowdedness'] != 'busy']
        return rec_rests, 'This restaurant is not busy, so you can choose your own seats.'
    if req_consequent == 'children':
        rec_rests = rec_rests[rec_rests['length_stay'] != 'long stay']
        return rec_rests, 'This restaurants can be visited with children, since you do not have to stay long.'
    if req_consequent == 'not children' or req_consequent == 'no children':
        rec_rests = rec_rests[rec_rests['length_stay'] == 'long stay']
        return rec_rests, 'Guests usually spend a long time in this restaurant, which is not recommended when taking children.'
    if req_consequent == 'romantic':
        rec_rests = rec_rests[rec_rests['length_stay'] == 'long stay']
        return rec_rests, 'This restaurant is romantic because you can stay a long time.'
    if req_consequent == 'not romantic':
        rec_rests = rec_rests[rec_rests['length_stay'] == 'busy']
        return rec_rests, 'This restaurant is not romantic because it is busy.'
    return rec_rests, 'This restaurant should be fine.'

def pop_recommendation(recommendations: pd.DataFrame):
    # print(recommendations)
    if len(recommendations) < 1:
        return "no alternative possible", recommendations
    selected_restaurant = recommendations.sample(n=1, random_state=5).iloc[0]
    recommendations.drop(selected_restaurant.name, inplace=True)
    return selected_restaurant["restaurantname"], recommendations

def get_restaurant_info(restaurants_df: pd.DataFrame, restaurantname: str) -> dict:
    """Given a restaurant name, return its information as a dictionary.

    Returns: dictionary with keys 'restaurantname', 'area', 'pricerange', 'food', 'phone', 'addr', 'postcode'.
    """
    if restaurantname not in restaurants_df["restaurantname"].to_list():
        raise ValueError(f'Could not find {restaurantname} in database')
    return restaurants_df[restaurants_df["restaurantname"]==restaurantname].to_dict(orient='records')[0]

def test_uer():
    restaurant_data = pd.read_csv('Data/restaurant_info.csv')
    food_quality_vals = ['good food', 'mediocre food', 'bad food']
    crowdedness_vals = ['busy', 'quiet']
    length_stay_vals = ['long stay', 'medium stay', 'short stay']
    restaurant_data['food_quality'] = np.random.choice(food_quality_vals, restaurant_data.shape[0])
    restaurant_data['crowdedness'] = np.random.choice(crowdedness_vals, restaurant_data.shape[0])
    restaurant_data['length_stay'] = np.random.choice(length_stay_vals, restaurant_data.shape[0])

    while True:
        utterance = input("Hello, how can I help you?\nYour utterance ('1' to quit): ").lower()
        if utterance == '1':
            break
        
        # recommend a restaurant based on food, area, and pricerange
        info_dict = info_in_utterance(utterance, restaurant_data)
        recommendation, alternatives = provide_recommendations(restaurants_df=restaurant_data, req_food=info_dict["food"], req_area=info_dict["area"], req_pricerange=info_dict["pricerange"])
        print("Recommended restaurant:", recommendation)
        print("Recommendation info:", get_restaurant_info(restaurant_data, recommendation))  # type: ignore
        print('Alternatives dataframe:', alternatives)

        # further refine recommendation based on user preference
        preference = input('Do you have an additional preference?\n')
        recommendations, output_str = preference_reasoning(restaurant_data, preference)
        recommendation = recommendations.sample(n=1)
        print('Recommended restaurant:', recommendation['restaurantname'], f'\n{output_str}')

# test_uer()