import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # My implementation starts here.
    # For each row in the test set, score each model on the test set and store the result
    # in the probabilities list.
    num_rows = test_set.num_items
    hmm_data = test_set.get_all_Xlengths()

    for rowid in range(num_rows):
        test_data_x = hmm_data[rowid][0]
        seq_length = hmm_data[rowid][1]

        model_probs_for_row = {}
        best_probability = float("-Inf")
        best_guess = None

        for aword in models:
            # Score this model on the test sequence.
            logL = float("-Inf")
            try:
                logL = models[aword].score(test_data_x, seq_length)
            except Exception as e:
                pass
            finally:
                model_probs_for_row[aword] = logL
                if logL > best_probability:
                    best_probability = logL
                    best_guess = aword

        probabilities.append(model_probs_for_row)
        guesses.append(best_guess)

    return probabilities, guesses

