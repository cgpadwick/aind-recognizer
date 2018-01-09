import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
eps = 1e-20

asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(df_means['left-x'])) / (asl.df['speaker'].map(df_std['left-x']) + eps)
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(df_means['left-y'])) / (asl.df['speaker'].map(df_std['left-y']) + eps)
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x'])) / (asl.df['speaker'].map(df_std['right-x']) + eps)
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y'])) / (asl.df['speaker'].map(df_std['right-y']) + eps)

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']


def get_theta(y, x):
    # np.arctan2 expects the first argument to be the y coordinate, but the instructions say to swap the
    # x and y coordinates.
    return np.arctan2(x, y)


asl.df['polar-rr'] = np.sqrt(asl.df['grnd-rx'] ** 2 + asl.df['grnd-ry'] ** 2)
asl.df['polar-rtheta'] = get_theta(asl.df['grnd-ry'], asl.df['grnd-rx'])
asl.df['polar-lr'] = np.sqrt(asl.df['grnd-lx'] ** 2 + asl.df['grnd-ly'] ** 2)
asl.df['polar-ltheta'] = get_theta(asl.df['grnd-ly'], asl.df['grnd-lx'])

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

asl.df['delta-rx'] =  asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] =  asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] =  asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] =  asl.df['left-y'].diff().fillna(0)

features_custom = ['scaled-rx', 'scaled-ry', 'scaled-lx', 'scaled-ly']

data_min = asl.df.groupby('speaker').min()
data_max = asl.df.groupby('speaker').max()

asl.df['scaled-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(data_min['right-x'])) / \
(asl.df['speaker'].map(data_max['right-x']) - asl.df['speaker'].map(data_min['right-x']) + eps)

asl.df['scaled-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(data_min['right-y'])) / \
(asl.df['speaker'].map(data_max['right-y']) - asl.df['speaker'].map(data_min['right-y']) + eps)

asl.df['scaled-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(data_min['left-x'])) / \
(asl.df['speaker'].map(data_max['left-x']) - asl.df['speaker'].map(data_min['left-x']) + eps)

asl.df['scaled-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(data_min['left-y'])) / \
(asl.df['speaker'].map(data_max['left-y']) - asl.df['speaker'].map(data_min['left-y']) + eps)


from my_model_selectors import SelectorConstant

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorConstant)
print("Number of word models returned = {}".format(len(models)))

test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))

from my_recognizer import recognize
from asl_utils import show_errors

# TODO Choose a feature set and model selector
features = features_ground # change as needed
model_selector = SelectorConstant # change as needed

# TODO Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)