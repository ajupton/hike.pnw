# App necessities
from flask import Flask, render_template, request
import requests
# Data manipulation
import pandas as pd
import numpy as np
# Database connections
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
# lightfm hybrid recommendation algorithm
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
# For creating sparse matrices
from scipy.sparse import coo_matrix, csc_matrix
from scipy import sparse


pd.options.display.max_columns=25

#Initialize app
app = Flask(__name__, static_url_path='/static')

#Standard home page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

#After the user hits submit, the index page redirects to trail_recommendations.html
@app.route('/trail_recommendations', methods=['GET', 'POST'])
def recommendations():

    # Gather user input from ideal hike text selection
    user_input = request.form.getlist('user_feature_options[]')
    input_user_features = pd.DataFrame([" ".join(user_input)])

    # Gather user filters - location, feature1, feature2
    user_location = request.form['user_location']
    trail_feature_select1 = request.form['trail_feature_select1']
    trail_feature_select2 = request.form['trail_feature_select2']

    # Parse user input
    # Add ALL the features for the new user
    user_feature_new = pd.DataFrame()
    user_feature_new['epic'] = pd.np.where(input_user_features[0].str.contains('epic'), 1, 0)
    user_feature_new['snow'] = pd.np.where(input_user_features[0].str.contains('snow'), 1, 0)
    user_feature_new['flat'] = pd.np.where(input_user_features[0].str.contains('flat'), 1, 0)
    user_feature_new['challenging'] = pd.np.where(input_user_features[0].str.contains('challenging'), 1, 0)
    user_feature_new['long'] = pd.np.where(input_user_features[0].str.contains('long'), 1, 0)
    user_feature_new['beach'] = pd.np.where(input_user_features[0].str.contains('beach'), 1, 0)
    user_feature_new['beautiful'] = pd.np.where(input_user_features[0].str.contains('beautiful'), 1, 0)
    user_feature_new['scenic'] = pd.np.where(input_user_features[0].str.contains('scenic'), 1, 0)
    user_feature_new['amazing'] = pd.np.where(input_user_features[0].str.contains('amazing'), 1, 0)
    user_feature_new['awesome'] = pd.np.where(input_user_features[0].str.contains('awesome'), 1, 0)
    user_feature_new['gorgeous'] = pd.np.where(input_user_features[0].str.contains('gorgeous'), 1, 0)
    user_feature_new['fun'] = pd.np.where(input_user_features[0].str.contains('fun'), 1, 0)
    user_feature_new['peaceful'] = pd.np.where(input_user_features[0].str.contains('peaceful'), 1, 0)
    user_feature_new['wonderful'] = pd.np.where(input_user_features[0].str.contains('wonderful'), 1, 0)
    user_feature_new['pretty'] = pd.np.where(input_user_features[0].str.contains('pretty'), 1, 0)
    user_feature_new['cool'] = pd.np.where(input_user_features[0].str.contains('cool'), 1, 0)
    user_feature_new['river'] = pd.np.where(input_user_features[0].str.contains('river'), 1, 0)
    user_feature_new['scenery'] = pd.np.where(input_user_features[0].str.contains('scenery'), 1, 0)
    user_feature_new['incredible'] = pd.np.where(input_user_features[0].str.contains('incredible'), 1, 0)
    user_feature_new['spectacular'] = pd.np.where(input_user_features[0].str.contains('spectacular'), 1, 0)
    user_feature_new['wildflowers'] = pd.np.where(input_user_features[0].str.contains('wildflowers'), 1, 0)
    user_feature_new['breathtaking'] = pd.np.where(input_user_features[0].str.contains('breathtaking'), 1, 0)
    user_feature_new['water'] = pd.np.where(input_user_features[0].str.contains('water'), 1, 0)
    user_feature_new['quiet'] = pd.np.where(input_user_features[0].str.contains('quiet'), 1, 0)
    user_feature_new['paved'] = pd.np.where(input_user_features[0].str.contains('paved'), 1, 0)
    user_feature_new['fantastic'] = pd.np.where(input_user_features[0].str.contains('fantastic'), 1, 0)
    user_feature_new['short'] = pd.np.where(input_user_features[0].str.contains('|'.join(['short', 'quick'])), 1, 0)
    user_feature_new['recommended'] = pd.np.where(input_user_features[0].str.contains('|'.join(['recommend','recommended'])), 1, 0)
    user_feature_new['mountain_views'] = pd.np.where(input_user_features[0].str.contains('|'.join(['mountain','mountains'])), 1, 0)
    user_feature_new['lake'] = pd.np.where(input_user_features[0].str.contains('|'.join(['lake', 'lakes'])), 1, 0)
    user_feature_new['forest'] = pd.np.where(input_user_features[0].str.contains('|'.join(['forest', 'tree','trees', 'woods'])), 1, 0)
    user_feature_new['lovely'] = pd.np.where(input_user_features[0].str.contains('|'.join(['lovely', 'loved','love'])), 1, 0)
    user_feature_new['dog_friendly'] = pd.np.where(input_user_features[0].str.contains('|'.join(['dog', 'dogs','doggy', 'pup', 'puppy'])), 1, 0)
    user_feature_new['family_friendly'] = pd.np.where(input_user_features[0].str.contains('|'.join(['kid', 'kids','child','family', 'children'])), 1, 0)
    user_feature_new['relaxing'] = pd.np.where(input_user_features[0].str.contains('|'.join(['relaxing', 'relaxed'])), 1, 0)
    user_feature_new['beginnger_friendly'] = pd.np.where(input_user_features[0].str.contains('|'.join(['easy', 'beginner'])), 1, 0)
    user_feature_new['experts_only'] = pd.np.where(input_user_features[0].str.contains('|'.join(['expert', 'hard','difficult', 'tough'])), 1, 0)
    user_feature_new['waterfalls'] = pd.np.where(input_user_features[0].str.contains('|'.join(['waterfall', 'waterfalls','falls'])), 1, 0)
    user_feature_new.rename(index={0: 'new_user'})
    user_feature_new.index.names = ['review_author']

    # Make connection to database
    # Database name
    dbname = 'pnw_hike'

    # Set postgres username
    username = 'andy'

    ## Using an engine to connect to the postgres db
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database = dbname, user = username)


    # User features
    user_features_query = """
    SELECT * FROM user_features;
    """
    user_features_from_sql = pd.read_sql_query(user_features_query, con, index_col='review_author')

    # Trail features raw
    trail_reviews_raw_query = """
    SELECT * FROM trail_reviews_raw;
    """
    trail_reviews_raw_from_sql = pd.read_sql_query(trail_reviews_raw_query, con, index_col="index")

    # Trail urls and filtering info
    trail_urls_info_query = """
    SELECT * FROM trail_urls_info;
    """
    trail_urls_info = pd.read_sql_query(trail_urls_info_query,con, index_col="index")

    # User features
    user_features_df = user_features_from_sql.drop(["index", "review_text", "clean_review"], axis = 1)
    user_features = user_features_df.fillna(0)

    # Trail features filling blanks with 0
    trail_features = trail_reviews_raw_from_sql.fillna(0)

    # Convert user-feature space to sparse matrix
    user_features = sparse.csr_matrix(user_features.values)

    # Bring in some helper functions
    # These functions are adapted from the recsys cookbook
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
              .sum().unstack().reset_index(). \
              fillna(0).set_index(user_col)
      if norm:
          interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
      return interactions

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

    # Create a large sparse dataframe of extant user reviews/ratings
    interactions = create_interaction_matrix(trail_reviews_raw_from_sql, user_col='review_author', item_col='trail_name', rating_col='review_rating', norm=False, threshold=None)

    # Align users in the interaction and user matrices due to dropping some trails ----FIX THIS LATER!!
    # Identify which users are in the interaction matrix and not in user feature space
    key_diff = set(interactions.index).difference(user_features_from_sql.index)
    where_diff = interactions.index.isin(key_diff)

    # Filter interactions based on users present in user features
    interactions = interactions.loc[~interactions.index.isin(interactions[where_diff].index)]

    # Convert sparse dataframe into a sparse matrix
    interactions_matrix = sparse.csr_matrix(interactions.values)

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

    # Prep for trail dict
    trail_urls = trail_urls_info[['trail_name', 'trail_url']]

    # Convert new user features to a sparse matrix
    user_feature_new_sparse = sparse.csr_matrix(user_feature_new.values)

    def concatenate_csc_matrices_by_columns(matrix1, matrix2):
      '''
      Function to horizontally stack non-2D/non-coo-matrices because
      hstack is unhappy with my matrices
      '''
      new_data = np.concatenate((matrix1.data, matrix2.data))
      new_indices = np.concatenate((matrix1.indices, matrix2.indices))
      new_ind_ptr = matrix2.indptr + len(matrix1.data)
      new_ind_ptr = new_ind_ptr[1:]
      new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

      return csc_matrix((new_data, new_indices, new_ind_ptr))

    ## Combine new user-feature sparse matrix with current users' sparse matrix
    new_user_features = concatenate_csc_matrices_by_columns(user_feature_new_sparse, user_features)

    # Incorporate new user's selections into the interaction matrix
    interactions_new_user_df = pd.DataFrame().reindex_like(interactions).iloc[0:0]
    interactions_new_user_df.loc["new_user"] = 0
    new_interactions_df = pd.concat([interactions_new_user_df, interactions])
    interactions_new_user = sparse.csr_matrix(interactions_new_user_df.values)
    new_interactions_matrix = concatenate_csc_matrices_by_columns(interactions_new_user, interactions_matrix)

    # Make trail dict
    trails_in_interaction_matrix = pd.DataFrame(interactions_new_user_df.columns.T)
    trail_dict_prep = trails_in_interaction_matrix.merge(trail_urls, on='trail_name')

    # Add unique identifier to trail dict
    trail_dict_prep['trail_id'] = trail_dict_prep.index+1

    # Make trail dict
    trails_dict = create_trail_dict(trail_dict_prep, id_col = 'trail_name', name_col = 'trail_id')

    # With the new interactions df we can defined a user dictionary
    user_dict = create_user_dict(interactions = new_interactions_df)

    # Run model with new user features and interactions
    NUM_THREADS = 1 # I can only support one thread on my machine :(
    NUM_COMPONENTS = 30
    NUM_EPOCHS = 3
    ITEM_ALPHA = 1e-6

    # Let's fit a WARP model: these generally have the best performance.
    model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS)

    # Run 3 epochs
    model = model.fit(interactions_new_user, user_features=new_user_features, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

    def new_user_recommendation(model, interactions, trail_urls_info, user_dict, trail_dict, user_location = user_location, trail_feature_select1 = trail_feature_select1, trail_feature_select2 = trail_feature_select2, user_id = "new_user", threshold = 3, nrec_items = 300):
      '''
      Function to produce user recommendations
      Required Input -
          - model = Trained model
          - interactions = dataset used for training the model
          - trail_urls = urls to return from predictions
          - user_id = user ID for which we need to generate recommendation
          - user_dict = Dictionary type input containing interaction_index as key and user_id as value
          - trail_dict = Dictionary type input containing trail_id as key and item_name as value
          - threshold = value above which the rating is favorable in new interaction matrix
          - nrec_items = Number of output recommendation needed
          - location = location filter selection
          - trail_feature_select1 (1-3) = trail filter feature selections
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

    # Run the model
    trail_names, trail_overviews, trail_urls, card_image_urls = new_user_recommendation(model, new_interactions_df, user_id="new_user", trail_urls_info=trail_urls_info, user_location=user_location, trail_feature_select1=trail_feature_select1, trail_feature_select2=trail_feature_select2, user_dict=user_dict, trail_dict=trails_dict, nrec_items=1000, threshold=4)

    # Change 'e' if selected
    if user_location == 'e':
      user_location = "all of the Pacific Northwest"

    return render_template('trail_recommendations.html', trail_names = trail_names, trail_overviews = trail_overviews, trail_urls = trail_urls, card_image_urls = card_image_urls, trail_feature_select1 = trail_feature_select1, trail_feature_select2 = trail_feature_select2, user_location = user_location, input_user_features = user_input)

if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=8080, debug=True)
