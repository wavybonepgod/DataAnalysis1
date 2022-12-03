import numpy as np
import os.path
from enum import Enum
import re
import math


class LetterTypes(Enum):
    Ham = 1,
    Spam = 2


delete_non_characters_regex = re.compile('[^a-zA-Z ]')
delete_multi_spaces = re.compile(' {2,}')


def init_data_sets():
    ham_path = './Resources/ham'
    spam_path = './Resources/spam'

    ham_letters = [name for name in os.listdir(ham_path)]
    spam_letters = [name for name in os.listdir(spam_path)]
    ham_letters_count = len(ham_letters)
    spam_letters_count = len(spam_letters)

    total_letters_count = ham_letters_count + spam_letters_count

    letters = np.empty(total_letters_count, dtype=np.dtype([('text', object), ('type', LetterTypes)]))

    for index, letterName in enumerate(ham_letters):
        file = open(ham_path + "/" + letterName, "r", encoding='utf8', errors='ignore')
        letters[index] = (file.read(), LetterTypes.Ham)
        file.close()

    for index, letterName in enumerate(spam_letters):
        file = open(spam_path + "/" + letterName, "r", encoding='utf8', errors='ignore')
        letters[ham_letters_count + index] = (file.read(), LetterTypes.Spam)
        file.close()

    np.random.shuffle(letters)

    train_set_size = int(total_letters_count * 0.6)
    validation_set_size = int(total_letters_count * 0.2)
    test_set_size = total_letters_count - train_set_size - validation_set_size

    return letters[: train_set_size], \
           letters[train_set_size + 1: train_set_size + validation_set_size], \
           letters[-test_set_size:]


def get_text_words(text):
    text = delete_non_characters_regex.sub("", text)
    trimmed_text = delete_multi_spaces.sub(" ", text).strip()
    words = trimmed_text.split(" ")

    return words


def build_words_dictionary(letters_set):
    words = {}

    for letter in letters_set:
        letter_words = get_text_words(letter['text'])
        for word in letter_words:
            if word in words:
                temp = words.get(word)

                if letter['type'] == LetterTypes.Ham:
                    ham_count = temp[0] + 1
                    spam_count = temp[1]
                else:
                    ham_count = temp[0]
                    spam_count = temp[1] + 1

                words.update({word: [ham_count, spam_count]})
            else:
                if letter['type'] == LetterTypes.Ham:
                    ham_count = 1
                    spam_count = 0
                else:
                    ham_count = 0
                    spam_count = 1
                words.update({word: [ham_count, spam_count]})

    return words


def build_probability_dictionary(words, total_ham, total_spam):
    probabilities = {}

    for word in words:
        counters = words.get(word)

        probabilities \
            .update({word: [(float(counters[0]) + 1) / (total_ham + 2), (float(counters[1]) + 1) / (total_spam + 2)]})

    return probabilities


def detect_spam(letters, probabilities, threshold):
    run_correct_count = 0
    run_second_type_errors_count = 0
    run_first_type_errors_count = 0

    for letter in letters:
        words = get_text_words(letter['text'])

        ham_words_prob = 0.0
        spam_words_prob = 0.0

        for word in words:
            if word in probabilities:
                prob = probabilities.get(word)
                ham_words_prob += math.log(prob[0])
                spam_words_prob += math.log(prob[1])
            else:
                ham_words_prob += math.log(threshold)
                spam_words_prob += math.log(threshold)

        ham_words_prob = math.exp(ham_words_prob)
        spam_words_prob = math.exp(spam_words_prob)

        if (spam_words_prob + ham_words_prob) == 0.0:
            spam_prob = 0.5
        else:
            spam_prob = spam_words_prob / (spam_words_prob + ham_words_prob)

        if spam_prob > threshold:
            if letter['type'] == LetterTypes.Spam:
                run_correct_count += 1
            else:
                run_first_type_errors_count += 1
        else:
            if letter['type'] == LetterTypes.Ham:
                run_correct_count += 1
            else:
                run_second_type_errors_count += 1

    return run_correct_count, run_first_type_errors_count, run_second_type_errors_count


def main():
    train_set, validation_set, test_set = init_data_sets()

    train_ham_count = len(np.where(train_set['type'] == LetterTypes.Ham)[0])
    train_spam_count = len(np.where(train_set['type'] == LetterTypes.Spam)[0])

    words_dictionary = build_words_dictionary(train_set)
    probability_dictionary = build_probability_dictionary(words_dictionary, train_ham_count, train_spam_count)
    best_detection_value = 0.0
    best_detection_threshold = 0.0

    for detection_threshold in np.arange(0.01, 1, 0.01):
        correct_count, first_type_errors_count, second_type_errors_count = \
            detect_spam(validation_set, probability_dictionary, detection_threshold)

        detection_value = correct_count / (correct_count + first_type_errors_count + second_type_errors_count)
        print("Threshold: " + str(detection_threshold)
              + ", detection: " + str(detection_value)
              + ", first type error: " + str(first_type_errors_count)
              + ", second type error: " + str(second_type_errors_count)
              + ", total letters: " + str(correct_count + first_type_errors_count + second_type_errors_count))

        if best_detection_value < detection_value:
            best_detection_value = detection_value
            best_detection_threshold = detection_threshold

    print("Best threshold: " + str(best_detection_threshold) + ", detection: " + str(best_detection_value))

    test_correct_count, test_first_type_errors_count, test_second_type_errors_count = \
        detect_spam(test_set, probability_dictionary, best_detection_threshold)

    print("Test threshold: " + str(best_detection_threshold)
          + ", detection: " + str(
        test_correct_count / (test_correct_count + test_first_type_errors_count + test_second_type_errors_count))
          + ", first type error: " + str(test_first_type_errors_count)
          + ", second type error: " + str(test_second_type_errors_count)
          + ", total letters: " + str(
        test_correct_count + test_first_type_errors_count + test_second_type_errors_count))


if __name__ == "__main__":
    main()
