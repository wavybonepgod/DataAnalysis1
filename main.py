import helpers.routines as routines
import numpy as np
from models.letter_type import LetterTypes
from helpers.routines import build_words_dictionary
import spam_filters.naive_bayes_filter as bayes


def analyze_bayes_filter():
    best_detection_precision = 0.0
    best_detection_accuracy = 0.0

    best_detection_threshold = 0.0
    best_spam_probability = 0.0

    start_detection_threshold = 0.05
    start_spam_probability = 0.05

    bayes_filter = bayes.BayesSpamFilter(words_dictionary,
                                         train_ham_count,
                                         train_spam_count,
                                         start_detection_threshold,
                                         start_spam_probability)

    print("Started bayes filter validation")

    for detection_threshold in np.arange(start_detection_threshold, 1, 0.05):
        for spam_probability in np.arange(start_spam_probability, 1, 0.05):
            bayes_filter.set_threshold(detection_threshold)
            bayes_filter.set_spam_probability(spam_probability)

            true_positive, false_positive, true_negative, false_negative \
                = bayes_filter.detect_spam(validation_set)

            accuracy \
                = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
            precision = true_positive / (true_positive+false_positive)
            recall = true_positive / (true_positive+false_negative)

            print("{:0.2f}".format(detection_threshold)
                  + ",{:0.2f}".format(spam_probability)
                  + "," + str(false_positive)
                  + "," + str(false_negative)
                  + ",{:0.6f}".format(accuracy)
                  + ",{:0.6f}".format(precision)
                  + ",{:0.6f}".format(recall))

            if best_detection_precision < precision:
                best_detection_precision = precision
                best_detection_accuracy = accuracy
                best_detection_threshold = detection_threshold
                best_spam_probability = spam_probability
            elif best_detection_precision == precision and best_detection_accuracy > accuracy:
                best_detection_precision = precision
                best_detection_accuracy = accuracy
                best_detection_threshold = detection_threshold
                best_spam_probability = spam_probability

    print("Best threshold: " + "{:0.2f}".format(best_detection_threshold)
          + ", best spam probability: " + "{:0.2f}".format(best_spam_probability))

    print("Validation finished")

    print("Started bayes filter test")

    bayes_filter.set_threshold(best_detection_threshold)
    bayes_filter.set_spam_probability(best_spam_probability)
    true_positive, false_positive, true_negative, false_negative = \
        bayes_filter.detect_spam(test_set)

    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    print("Test threshold: " + "{:0.2f}".format(best_detection_threshold)
          + ", spam probability: " + "{:0.2f}".format(best_spam_probability)
          + ", first type errors: " + str(false_positive)
          + ", second type errors: " + str(false_negative)
          + ", accuracy: " + "{:0.6f}".format(accuracy)
          + ", precision: " + "{:0.6f}".format(precision)
          + ", recall: " + "{:0.6f}".format(recall))

    print("Test finished")


train_set, validation_set, test_set = routines.init_data_sets()

train_ham_count = np.where(train_set['type'] == LetterTypes.ham)[0].size
train_spam_count = np.where(train_set['type'] == LetterTypes.spam)[0].size

words_dictionary = build_words_dictionary(train_set)

analyze_bayes_filter()
