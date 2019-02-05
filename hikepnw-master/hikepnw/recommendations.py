import pandas as pd
import numpy as np

# Recommendation engine helper functions
# These functions are adapted from the recsys cookbook

def new_user_recommendation(model, interactions, trail_urls_info, user_dict, trail_dict, user_location, trail_feature_select1, trail_feature_select2, user_id = "new_user", threshold = 3, nrec_items = 300):
    '''
    Function to produce user recommendations
    Required Input -
        - model = Trained LightFM model
        - interactions = same dataset used for training the model - user/trail ratings sparse matrix
        - trail_urls_info = urls, descriptions, and card images to return from predictions
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - trail_dict = Dictionary type input containing trail_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendations to be filtered
        - location = location filter selection
        - trail_feature_select1 (1-2) = trail filter feature selections
    Expected Output -
        - Prints list of trails the given user has already rated
        - Prints list of N recommended trails which new user hopefully will be interested in
    '''
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index) \
                                 .sort_values(ascending=False))

    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    return_score_df = pd.DataFrame({'trail_name':return_score_list})
    return_score_df = return_score_df.merge(trail_urls_info, on="trail_name")
    # Filter for location. 'e' is for everything => Show me all PNW trails
    if user_location != 'e':
        return_score_df = return_score_df.loc[return_score_df["location"] == user_location]
    # Filter for feature 1
    return_score_df = return_score_df.loc[return_score_df[trail_feature_select1] == 1]
    # Filter for feature 2
    return_score_df = return_score_df.loc[return_score_df[trail_feature_select2] == 1]
    # Only include top 15 selection
    return_score_df = return_score_df.head(15)
    # Parse dataframe into lists for return homepage
    trail_names = list(return_score_df["trail_name"])
    trail_overviews = list(return_score_df["overview"])
    trail_urls = list(return_score_df["trail_url"])
    card_image_urls = list(return_score_df["card_image_url"])

    return trail_names, trail_overviews, trail_urls, card_image_urls


def create_trail_dict(df,id_col,name_col):
  '''
  Function to create an item dictionary based on their item_id and item name
  Required Input -
      - df = Pandas dataframe with Item information
      - id_col = Column name containing unique identifier for an item
      - name_col = Column name containing name of the item
  Expected Output -
      item_dict = Dictionary type output containing item_id as key and item_name as value
  '''
  item_dict ={}
  for i in range(df.shape[0]):
      item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
  return item_dict


def create_user_dict(interactions):
  '''
  Function to create a user dictionary based on their index and number in interaction dataset
  Required Input -
      interactions - dataset create by create_interaction_matrix
  Expected Output -
      user_dict - Dictionary type output containing interaction_index as key and user_id as value
  '''
  user_id = list(interactions.index)
  user_dict = {}
  counter = 0
  for i in user_id:
      user_dict[i] = counter
      counter += 1
  return user_dict

def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
  '''
  Function to create an interaction matrix dataframe from transactional type interactions
  Required Input -
      - df = Pandas DataFrame containing user-item interactions
      - user_col = column name containing user's identifier
      - item_col = column name containing item's identifier
      - rating col = column name containing user feedback on interaction with a given item
      - norm (optional) = True if a normalization of ratings is needed
      - threshold (required if norm = True) = value above which the rating is favorable
  Expected output -
      - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
  '''
  interactions = df.groupby([user_col, item_col])[rating_col] \
          .sum().unstack(level=0, fill_value=0.0).T.reset_index() \
          .set_index(user_col)
  if norm:
      interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
  return interactions
