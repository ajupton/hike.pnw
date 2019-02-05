import pandas as pd
import numpy as np


# Add ALL the features for the new user
def parse_input_descriptors(input_user_features):
    '''
    Function to parse input descriptors of ideal trails into a vector of user
    features.
    Required input -
        -input_user_features = list of user feature selections
    Expected output -
        -user_feature_new = pandas df of binary vector indicating user features
    '''
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
    return user_feature_new


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
