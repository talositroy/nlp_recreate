# import python modules -- this cell needs to be run again if you make changes to any of the files
import matplotlib.pyplot as plt
import numpy as np

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)


# print("There are {} sentences in the corpus.".format(len(data)))
# print("There are {} sentences in the training set.".format(len(data.training_set)))
# print("There are {} sentences in the testing set.".format(len(data.testing_set)))
#
# assert len(data) == len(data.training_set) + len(data.testing_set), \
#        "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"
#
# key = 'b100-38532'
# print("Sentence: {}".format(key))
# print("words:\n\t{!s}".format(data.sentences[key].words))
# print("tags:\n\t{!s}".format(data.sentences[key].tags))
#
# print("There are a total of {} samples of {} unique words in the corpus."
#       .format(data.N, len(data.vocab)))
# print("There are {} samples of {} unique words in the training set."
#       .format(data.training_set.N, len(data.training_set.vocab)))
# print("There are {} samples of {} unique words in the testing set."
#       .format(data.testing_set.N, len(data.testing_set.vocab)))
# print("There are {} words in the test set that are missing in the training set."
#       .format(len(data.testing_set.vocab - data.training_set.vocab)))
#
# assert data.N == data.training_set.N + data.testing_set.N, \
#        "The number of training + test samples should sum to the total number of samples"

# accessing words with Dataset.X and tags with Dataset.Y
# for i in range(2):
#     print("Sentence {}:".format(i + 1), data.X[i])
#     print()
#     print("Labels {}:".format(i + 1), data.Y[i])
#     print()

# use Dataset.stream() (word, tag) samples for the entire corpus
# print("\nStream (word, tag) pairs:\n")
# for i, pair in enumerate(data.stream()):
#     print("\t", pair)
#     if i > 5: break


def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.

    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    # TODO: Finish this function!
    count_sequences = 0
    dict = {}
    sequences_B = sequences_B.lower()


    for pair in enumerate(data.stream()):
        for tag in data.tagset:
            if (pair[1][1] == tag and pair[1][0].lower() == sequences_B):
                count_sequences += 1

    dict = {sequences_A: count_sequences}
    return dict
    # raise NotImplementedError


# Calculate C(t_i, w_i)
emission_counts = pair_counts('NOUN', 'time')
print(emission_counts)
assert len(emission_counts) == 12, \
       "Uh oh. There should be 12 tags in your dictionary."
assert max(emission_counts["NOUN"], key=emission_counts["NOUN"].get) == 'time', \
       "Hmmm...'time' is expected to be the most common NOUN."
HTML('<div class="alert alert-block alert-success">Your emission counts look good!</div>')
