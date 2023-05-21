# author: Arnoud De Jonge

CATEGORY_INT_TO_STRING = {0: 'category_bakeries', 1: 'category_coffee_&_tea', 2: 'category_restaurants', 3: 'category_food',
                          4: 'category_ice_cream_&_frozen_yogurt', 5: 'category_burgers', 6: 'category_fast_food', 7: 'category_sandwiches',
                          8: 'category_american_(traditional)', 9: 'category_bars', 10: 'category_italian', 11: 'category_nightlife', 12: 'category_greek',
                          13: 'category_pubs', 14: 'category_food_trucks', 15: 'category_vietnamese', 16: 'category_breakfast_&_brunch', 17: 'category_diners',
                          18: 'category_delis', 19: 'category_japanese', 20: 'category_sushi_bars', 21: 'category_cafes', 22: 'category_wine_bars',
                          23: 'category_steakhouses', 24: 'category_asian_fusion', 25: 'category_hot_dogs', 26: 'category_seafood',
                          27: 'category_cocktail_bars', 28: 'category_chicken_wings', 29: 'category_pizza', 30: 'category_salad', 31: 'category_soup',
                          32: 'category_arts_&_entertainment', 33: 'category_specialty_food', 34: 'category_chinese', 35: 'category_event_planning_&_services',
                          36: 'category_caterers', 37: 'category_american_(new)', 38: 'category_sports_bars', 39: 'category_beer_bar',
                          40: 'category_gastropubs', 41: 'category_lounges', 42: 'category_convenience_stores', 43: 'category_venues_&_event_spaces',
                          44: 'category_juice_bars_&_smoothies', 45: 'category_shopping', 46: 'category_cajun/creole', 47: 'category_mexican',
                          48: 'category_french', 49: 'category_mediterranean', 50: 'category_wine_&_spirits', 51: 'category_beer', 52: 'category_barbeque',
                          53: 'category_chicken_shop', 54: 'category_thai', 55: 'category_bagels', 56: 'category_southern', 57: 'category_music_venues',
                          58: 'category_vegan', 59: 'category_soul_food', 60: 'category_desserts', 61: 'category_food_delivery_services',
                          62: 'category_caribbean', 63: 'category_tex-mex', 64: 'category_ethnic_food', 65: 'category_gluten-free',
                          66: 'category_latin_american', 67: 'category_comfort_food', 68: 'category_vegetarian', 69: 'category_indian', 70: 'category_buffets',
                          71: 'category_middle_eastern', 72: 'category_tacos', 73: 'category_cheesesteaks', 74: 'category_grocery'}

CATEGORY_STRING_TO_INT = {value: key for key, value in CATEGORY_INT_TO_STRING.items()}


def convert_category_to_string(genre: int):
    return CATEGORY_INT_TO_STRING[genre]


def convert_category_to_int(genre: str):
    return CATEGORY_STRING_TO_INT[genre]
