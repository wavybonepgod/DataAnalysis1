import numpy as np
import os.path
import re
from models.letter_type import LetterTypes


def init_data_sets():
    ham_path = './resources/ham'
    spam_path = './resources/spam'

    ham_letters = [name for name in os.listdir(ham_path)]
    spam_letters = [name for name in os.listdir(spam_path)]
    ham_letters_count = len(ham_letters)
    spam_letters_count = len(spam_letters)

    total_letters_count = ham_letters_count + spam_letters_count

    letters = np.empty(total_letters_count, dtype=np.dtype([('text', object), ('type', LetterTypes)]))

    for index, letterName in enumerate(ham_letters):
        with open(ham_path + "/" + letterName, "r", encoding='utf8', errors='ignore') as file:
            letters[index] = (file.read(), LetterTypes.ham)

    for index, letterName in enumerate(spam_letters):
        with open(spam_path + "/" + letterName, "r", encoding='utf8', errors='ignore') as file:
            letters[ham_letters_count + index] = (file.read(), LetterTypes.spam)

    np.random.shuffle(letters)

    train_set_size = int(total_letters_count * 0.6)
    validation_set_size = int(total_letters_count * 0.2)
    test_set_size = total_letters_count - train_set_size - validation_set_size

    return letters[: train_set_size], \
        letters[train_set_size + 1: train_set_size + validation_set_size], \
        letters[-test_set_size:]


def build_words_dictionary(letters_set):
    words = {}

    for letter in letters_set:
        letter_words = get_text_words(letter['text'])
        for word in letter_words:
            if word in words:
                temp = words.get(word)

                if letter['type'] == LetterTypes.ham:
                    ham_count = temp[0] + 1
                    spam_count = temp[1]
                else:
                    ham_count = temp[0]
                    spam_count = temp[1] + 1

                words.update({word: [ham_count, spam_count]})
            else:
                if letter['type'] == LetterTypes.ham:
                    ham_count = 1
                    spam_count = 0
                else:
                    ham_count = 0
                    spam_count = 1
                words.update({word: [ham_count, spam_count]})

    return words


def get_text_words(text):
    delete_non_characters_regex = re.compile('[^a-zA-Z ]')
    delete_multi_spaces = re.compile(' {2,}')

    text = delete_non_characters_regex.sub("", text)
    trimmed_text = delete_multi_spaces.sub(" ", text).strip()
    words = trimmed_text.split(" ")

    return words
