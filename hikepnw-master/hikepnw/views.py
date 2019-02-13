# App necessities
from flask import Flask, render_template, request
import requests
from hikepnw import app
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
# Recommendation helper functions
from hikepnw.manipulation import parse_input_descriptors, concatenate_csc_matrices_by_columns
from hikepnw.recommendations import new_user_recommendation, create_trail_dict, create_interaction_matrix, create_user_dict

#Standard home page
@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

#After the user hits submit, the index page redirects to trail_recommendations.html
@app.route('/trail_recommendations', methods=['GET', 'POST'])
def recommendations():
    """
    Render the trail_recommendations.html page

    Args:
        Nothing

    Returns:
        the trail_recommendations.html template, this includes hiking trails
        recommendations based on user-input. Up to 10 trails are provided.
        Trail options are presented in cards that include a photo taken of the
        trail, a short description of the trail, and a link to the trail
        profile page on AllTrails.com
    """
    # Gather user input from ideal hike text selection
    user_input = request.form.getlist('user_feature_options[]')
    input_user_features = pd.DataFrame([" ".join(user_input)])

    # Gather user filters - location, feature1, feature2
    user_location = request.form['user_location']
    trail_feature_select1 = request.form['trail_feature_select1']
    trail_feature_select2 = request.form['trail_feature_select2']

    # Parse user input
    user_feature_new = parse_input_descriptors(input_user_features)

    # Make connection to database
    # Database name
    dbname = 'pnw_hike'

    # Set postgres username
    username = 'ubuntu'

    ## Using an engine to connect to the postgres db
    engine = create_engine('postgres://%s:insight@localhost/%s'%(username, dbname), paramstyle="format")

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database = dbname, user = username, password = 'insight', port = 5432)

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

    # Create a large sparse dataframe of extant user reviews/ratings
    interactions = create_interaction_matrix(trail_reviews_raw_from_sql, user_col='review_author', item_col='trail_name', rating_col='review_rating', norm=False, threshold=None)

    # Align users in the interaction and user matrices due to dropping some trails
    # Identify which users are in the interaction matrix and not in user feature space
    key_diff = set(interactions.index).difference(user_features_from_sql.index)
    where_diff = interactions.index.isin(key_diff)

    # Filter interactions based on users present in user features
    interactions = interactions.loc[~interactions.index.isin(interactions[where_diff].index)]

    # Convert sparse dataframe into a sparse matrix
    interactions_matrix = sparse.csr_matrix(interactions.values)

    # Prep for trail dict
    trail_urls = trail_urls_info[['trail_name', 'trail_url']]

    # Convert new user features to a sparse matrix
    user_feature_new_sparse = sparse.csr_matrix(user_feature_new.values)

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
    NUM_THREADS = 4 # The t2.xlarge instance supports up to 4 cores, we'll use all 4 here
    NUM_COMPONENTS = 30
    NUM_EPOCHS = 5
    ITEM_ALPHA = 1e-6

    # Let's train a WARP model: these generally have the best performance.
    model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS, random_state=15)

    # Fit model
    model = model.fit(interactions=new_interactions_matrix, user_features=new_user_features,
                      epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

    # Run the model
    trail_names, trail_overviews, trail_urls, card_image_urls = new_user_recommendation(model,
                                                                                        new_interactions_df,
                                                                                        user_id="new_user",
                                                                                        trail_urls_info=trail_urls_info,
                                                                                        user_location=user_location,
                                                                                        trail_feature_select1=trail_feature_select1,
                                                                                        trail_feature_select2=trail_feature_select2,
                                                                                        user_dict=user_dict, trail_dict=trails_dict,
                                                                                        nrec_items=1500,
                                                                                        threshold=4)

    # Change 'e' if selected
    if user_location == 'e':
      user_location = "all of the Pacific Northwest"

    return render_template('trail_recommendations.html',
                            trail_names = trail_names,
                            trail_overviews = trail_overviews,
                            trail_urls = trail_urls,
                            card_image_urls = card_image_urls,
                            trail_feature_select1 = trail_feature_select1,
                            trail_feature_select2 = trail_feature_select2,
                            user_location = user_location,
                            input_user_features = user_input)

@app.route('/about')
def about():
    ''' About page
    '''
    return render_template("about.html")
