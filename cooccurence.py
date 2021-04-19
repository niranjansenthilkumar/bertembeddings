import numpy as np
import nltk
from nltk import bigrams
import itertools
import pandas as pd
import loader
import collections, numpy
import pickle
 
def generate_co_occurrence_matrix(corpus):
    vocab = set(corpus)
    vocab = list(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
 
    bi_grams = list(bigrams(corpus))
 
    bigram_freq = nltk.FreqDist(bi_grams).most_common(len(bi_grams))
 
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
 
    for bigram in bigram_freq:
        current = bigram[0][1]
        previous = bigram[0][0]
        count = bigram[1]
        pos_current = vocab_index[current]
        pos_previous = vocab_index[previous]
        co_occurrence_matrix[pos_current][pos_previous] = count
    co_occurrence_matrix = np.matrix(co_occurrence_matrix)
 
    return co_occurrence_matrix, vocab_index
 
 
def main():

    train_data = loader.load_data('dataset/train.csv', False)
    valid_data = loader.load_data('dataset/dev.csv', False)
    test_data = loader.load_data('dataset/test.csv', True)


    train_array = []
    for example in train_data:

        # print(example[1])

        first_arr = []
        second_arr = []
        
        for t in example[1]:
            innerarr = t.split()
            first_arr.append(innerarr)
            second_arr.append(innerarr)

        first_ending = example[2][0].split(" ")
        second_ending = example[2][1].split(" ")


        first_arr.append(first_ending)
        second_arr.append(second_ending)

        # Create one list using many lists
        data = list(itertools.chain.from_iterable(first_arr))
        matrix, vocab_index = generate_co_occurrence_matrix(data)

        second_data = list(itertools.chain.from_iterable(second_arr))
        second_matrix, second_vocab_index = generate_co_occurrence_matrix(second_data)
         
        data_matrix = pd.DataFrame(matrix, index=vocab_index,
                                     columns=vocab_index)


        # print(count)
        first_ending_val = numpy.sum(matrix)
        second_ending_val = numpy.sum(second_matrix)


        array = [first_ending_val, second_ending_val]

        train_array.append(array)


    valid_array = []
    for example in valid_data:

        # print(example[1])

        first_arr = []
        second_arr = []
        
        for t in example[1]:
            innerarr = t.split()
            first_arr.append(innerarr)
            second_arr.append(innerarr)

        first_ending = example[2][0].split(" ")
        second_ending = example[2][1].split(" ")


        first_arr.append(first_ending)
        second_arr.append(second_ending)

        # Create one list using many lists
        data = list(itertools.chain.from_iterable(first_arr))
        matrix, vocab_index = generate_co_occurrence_matrix(data)

        second_data = list(itertools.chain.from_iterable(second_arr))
        second_matrix, second_vocab_index = generate_co_occurrence_matrix(second_data)
         
        data_matrix = pd.DataFrame(matrix, index=vocab_index,
                                     columns=vocab_index)


        # print(count)
        first_ending_val = numpy.sum(matrix)
        second_ending_val = numpy.sum(second_matrix)


        array = [first_ending_val, second_ending_val]

        valid_array.append(array)


    test_array = []
    for example in test_data:

        # print(example[1])

        first_arr = []
        second_arr = []
        
        for t in example[1]:
            innerarr = t.split()
            first_arr.append(innerarr)
            second_arr.append(innerarr)

        first_ending = example[2][0].split(" ")
        second_ending = example[2][1].split(" ")


        first_arr.append(first_ending)
        second_arr.append(second_ending)

        # Create one list using many lists
        data = list(itertools.chain.from_iterable(first_arr))
        matrix, vocab_index = generate_co_occurrence_matrix(data)

        second_data = list(itertools.chain.from_iterable(second_arr))
        second_matrix, second_vocab_index = generate_co_occurrence_matrix(second_data)
         
        data_matrix = pd.DataFrame(matrix, index=vocab_index,
                                     columns=vocab_index)


        # print(count)
        first_ending_val = numpy.sum(matrix)
        second_ending_val = numpy.sum(second_matrix)


        array = [first_ending_val, second_ending_val]

        test_array.append(array)


    print(len(train_array))
    print(len(valid_array))
    print(len(test_array))

    pickle_out_train = open('train_coc.pickle', 'wb')
    pickle.dump(train_array, pickle_out_train)
    pickle_out_train.close()


    pickle_out_valid = open('valid_coc.pickle', 'wb')
    pickle.dump(valid_array, pickle_out_valid)
    pickle_out_valid.close()

    pickle_out_test = open('test_coc.pickle', 'wb')
    pickle.dump(test_array, pickle_out_test)
    pickle_out_test.close()

if __name__ == '__main__':
    main()