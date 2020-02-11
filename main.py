import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import tflearn
import numpy as np
import random as rd

import actions as ac

data_name = 'intents_asktime.json'
model_name = 'asktime.tflearn'


def convert_text(func_input):
    chars = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss'}
    for char in chars:
        func_input = func_input.replace(char, chars[char])
    return func_input


if __name__ == '__main__':
    with open(f'data/{data_name}') as file:
        intents_data = json.load(file)
        lanc_stemmer = LancasterStemmer()

        word_list = []
        tag_list = []
        word_pattern_list = []
        tag_total_list = []

        for intent in intents_data['intents']:
            for pattern in intent['patterns']:
                pattern_tokenized = nltk.word_tokenize(convert_text(pattern))
                word_list.extend(pattern_tokenized)
                word_pattern_list.append(pattern_tokenized)
                tag_total_list.append(intent['tag'])

            if intent['tag'] not in tag_list:
                tag_list.append(intent['tag'])

        word_list = [lanc_stemmer.stem(word.lower()) for word in word_list if word is not '?']
        word_list = sorted(list(set(word_list)))

        tag_list = sorted(tag_list)

        training_vector = []
        output_vector = []

        for i, pattern in enumerate(word_pattern_list):
            bag_training = np.zeros(len(word_list))

            words_stemmed = [lanc_stemmer.stem(word.lower()) for word in pattern]

            for stemmed_word in words_stemmed:
                for j, list_word in enumerate(word_list):
                    if list_word == stemmed_word:
                        bag_training[j] = 1

            output_row = np.zeros(len(tag_list))
            output_row[tag_list.index(tag_total_list[i])] = 1

            training_vector.append(bag_training)
            output_vector.append(output_row)

        training_vector = np.array(training_vector)
        output_vector = np.array(output_vector)

        net_large_layer = len(output_vector[0]) * 4
        net_small_layer = int(len(output_vector[0]) * 2)

        net = tflearn.input_data(shape=[None, len(training_vector[0])])
        net = tflearn.fully_connected(net, net_large_layer)
        net = tflearn.fully_connected(net, net_large_layer)
        net = tflearn.fully_connected(net, net_small_layer)
        net = tflearn.fully_connected(net, len(output_vector[0]), activation='softmax')
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        model.fit(training_vector, output_vector, n_epoch=500, batch_size=10, show_metric=True)
        model.save(f'models/{model_name}')


        def find_from_input(user_input, vocabulary):
            bag_input = np.zeros(len(word_list))

            input_text = nltk.word_tokenize(user_input)
            input_text = [lanc_stemmer.stem(word.lower()) for word in input_text]

            for input_word in input_text:
                for j, vocab_word in enumerate(vocabulary):
                    if vocab_word == input_word:
                        bag_input[j] = 1

            return np.array(bag_input)


        def chat():
            chat_quit = False
            print('Spreche mit dem Bot:')
            while not chat_quit:
                user_input = convert_text(input('Du: '))

                prediction_results = model.predict([find_from_input(user_input, word_list)])
                max_result_index = np.argmax(prediction_results)
                tag = tag_list[max_result_index]

                for tag_data in intents_data['intents']:
                    if tag_data['tag'] == tag:
                        responses = tag_data['responses']
                        if '.act' in responses[0]:
                            chat_quit = ac.action_manager(responses[0])
                        else:
                            print(rd.choice(responses))

        chat()
