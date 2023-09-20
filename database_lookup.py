import pandas as pd

restaurant_data = pd.read_csv("Data\\restaurant_info.csv", )

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

recommendation = provide_recommendations(restaurants_df=restaurant_data, req_food="italian", req_area='south')
print("Recommended restaurant:", recommendation)
print("Recommendation info:", get_restaurant_info(restaurant_data, recommendation))
