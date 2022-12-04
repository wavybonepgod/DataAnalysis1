import math

from helpers.routines import get_text_words
from models.letter_type import LetterTypes


class BayesSpamFilter:
    probability_dictionary = {}
    detection_threshold = 0.0
    spam_probability = 0.0

    def __init__(self, words_dictionary, total_ham, total_spam, initial_threshold, initial_spam_probability):
        self.detection_threshold = initial_threshold
        self.spam_probability = initial_spam_probability
        self.build_probability_dictionary(words_dictionary, total_ham, total_spam)

    def set_threshold(self, new_threshold):
        self.detection_threshold = new_threshold

    def set_spam_probability(self, spam_probability):
        self.spam_probability = spam_probability

    def build_probability_dictionary(self, words, total_ham, total_spam):
        for word in words:
            counters = words.get(word)

            self.probability_dictionary.update(
                {
                    word: [(float(counters[0]) + 1) / (total_ham + 2),
                           (float(counters[1]) + 1) / (total_spam + 2)]
                })

    def detect_spam(self, letters):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for letter in letters:
            words = get_text_words(letter['text'])

            ham_words_prob = 0.0
            spam_words_prob = 0.0

            for word in words:
                if word in self.probability_dictionary:
                    prob = self.probability_dictionary.get(word)
                    ham_words_prob += math.log(prob[0])
                    spam_words_prob += math.log(prob[1])
                else:
                    ham_words_prob += math.log(self.spam_probability)
                    spam_words_prob += math.log(self.spam_probability)

            ham_words_prob = math.exp(ham_words_prob)
            spam_words_prob = math.exp(spam_words_prob)

            if (spam_words_prob + ham_words_prob) == 0.0:
                spam_prob = 0.5
            else:
                spam_prob = spam_words_prob / (spam_words_prob + ham_words_prob)

            if spam_prob > self.detection_threshold:
                if letter['type'] == LetterTypes.spam:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if letter['type'] == LetterTypes.ham:
                    true_negative += 1
                else:
                    false_negative += 1

        return true_positive, false_positive, true_negative, false_negative
