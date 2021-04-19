from sklearn import svm
import loader
import pickle

unk = '<UNK>'

def make_vocab(data):
    vocab = {}
    for elt in data:
        for sentence in elt[1]:
            sentence = sentence[:len(sentence)-1].lower().split(' ')
            #print(sentence)
            for word in sentence:
                vocab[word] = None

    vocab[unk] = None
    return vocab


def make_indices(vocab):
    vocab_list = sorted(vocab)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    
    return word2index, index2word 


def convert_to_vector_representation(data, word2index, embeddings, mean):
    PADDING = [1000] * 50
    vectorized_data = []
    labels = []
    ID = []
    for elt in data:
        vector = []
        # vector = [0] * (len(word2index)*3)
        # for sentence in elt[1]:
        #     sentence = sentence[:len(sentence)-1].lower().split(' ')
        #     for word in sentence:
        #         index = word2index[word] if word in word2index else word2index[unk]
        #         vector[index] += 1

        ending1 = elt[2][0][:len(elt[2][0])-1].lower().split(' ')
        # for word in ending1:

        #     index = word2index[word] if word in word2index else word2index[unk]
        #     vector[index + len(word2index)] += 1

        ending2 = elt[2][1][:len(elt[2][1])-1].lower().split(' ')
        # for word in ending2:

        #     index = word2index[word] if word in word2index else word2index[unk]
        #     vector[index + 2*len(word2index)] += 1

        vector.append(len(ending1))
        vector.append(len(ending2))

        #print(vector)
        # for sentence in elt[1]:
        #     sentence = sentence[:len(sentence)-1].lower().split(' ')
        #     for i in range(0,15):
        #         if i < len(sentence):
        #             embed = embeddings[sentence[i]] if sentence[i] in embeddings else mean
        #             vector = vector + embed
        #         else:
        #             vector = vector + PADDING

        #         #print(vector)

        # for ending in [ending1,ending2]:
        #     for i in range(0,15):
        #         if i < len(ending):
        #             embed = embeddings[ending[i]] if ending[i] in embeddings else mean
        #             vector = vector + embed
        #         else:
        #             vector = vector + PADDING

        # dif = 4502 - len(vector)
        # if dif > 0:
        #     vector = [1000]*dif + vector

        vectorized_data.append(vector)
        labels.append(elt[3])
        ID.append(elt[0])

    return ID, vectorized_data, labels

def main():
    train_data = loader.load_data('data/train.csv')
    valid_data = loader.load_data('data/dev.csv')
    print('Data fetched')

    vocab = make_vocab(train_data)

    word2index, index2word = make_indices(vocab)

    pickle_in1 = open("dict.pickle", "rb")
    wordEmbeddings = pickle.load(pickle_in1)

    pickle_in2 = open("mean.pickle", "rb")
    meanVec = pickle.load(pickle_in2)

    train_id,train,train_labels = convert_to_vector_representation(train_data, word2index, wordEmbeddings, meanVec)
    valid_id,valid,valid_labels = convert_to_vector_representation(valid_data, word2index, wordEmbeddings, meanVec)

    classifier = svm.SVC()
    print('Training')
    classifier.fit(train,train_labels)

    print('Evaluating')
    print(classifier.predict(valid))
    print(classifier.score(valid,valid_labels))


