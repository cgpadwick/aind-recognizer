import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("Inf")
        best_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # This is in a try/except block because sometimes the score function
                # fails with an error.
                the_model = self.base_model(num_states)
                logL = the_model.score(self.X, self.lengths)
                bic_score = -2. * logL + float(num_states) * np.log(len(self.X))
                if bic_score < best_score:
                    best_score = bic_score
                    best_model = the_model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float("-Inf")
        best_model = None
        #total_num_words = len(self.words)
        for num_states in range(self.min_n_components, self.max_n_components + 1):

            # Compute the score for the model with the current word.
            the_model = self.base_model(num_states)
            try:
                this_word_logL = the_model.score(self.X, self.lengths)
            except Exception as e:
                #print('failed training for {}, num_states={}'.format(self.this_word, num_states))
                #print(e)
                continue

            #print('this_word_logL: {}'.format(this_word_logL))

            # Loop through all the words and compute the term SUM(log(P(X(all but i)).
            sum_vals = 0.0
            total_num_words = 0
            #print(self.words)
            for word in self.words.keys():
                if word != self.this_word:
                    X, lengths = self.hwords[word]
                    try:
                        word_score = the_model.score(X, lengths)
                        sum_vals += word_score
                        total_num_words += 1
                    except:
                        pass
                    #print('word_score: {} word:{}'.format(word_score, word))
            sum_vals /= float(total_num_words - 1.)

            #print('sum_vals: {}'.format(sum_vals))

            dic_score = this_word_logL - sum_vals
            #print('this_word: {}, num_states:{} sum_vals:{} this_word_logL:{} dic_score:{}'
            #      .format(self.this_word, num_states, sum_vals, this_word_logL, dic_score))
            if dic_score > best_score:
                best_score = dic_score
                best_model = the_model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-Inf")
        best_num_states = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            min_num_splits = 3
            split_method = KFold(n_splits=min_num_splits)
            sum_logL = 0.0

            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_X_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_X_lengths = combine_sequences(cv_test_idx, self.sequences)

                    # Fit a model to the training fold.
                    hmm_model = \
                        GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(train_X, train_X_lengths)

                    # Score the model on the test fold.
                    sum_logL += hmm_model.score(test_X, test_X_lengths)

            except Exception as e:
                pass
                #print(
                #'training failed for {} with num_states:{}'.format(self.this_word, num_states))

            avg_logL = sum_logL / float(min_num_splits)
            if avg_logL > best_score:
                best_score = avg_logL
                best_num_states = num_states

        return self.base_model(best_num_states)
