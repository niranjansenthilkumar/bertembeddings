import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path 
import time
from tqdm import tqdm
import loader

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
            super(FFNN, self).__init__()
            self.h = h
            self.W1 = nn.Linear(input_dim, h)
            self.W2 = nn.Linear(h, 2)
            self.activation = nn.Tanh()
            self.softmax = nn.LogSoftmax()
            self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        z1 = self.W1(input_vector)
        z1 = self.activation(z1)
        z2 = self.W2(z1)
        z2 = self.activation(z2)

        predicted_vector = self.softmax(z2)
        return predicted_vector


def make_vocab(data):
    print(len(data))
    vocab = {}
    for elt in data:
        for sentence in elt[1]:
            for word in sentence.lower().split(' '):
                vocab[word] = None

    vocab[unk] = None
    print(len(vocab))
    return vocab


def make_indices(vocab):
    vocab_list = sorted(vocab)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    
    return word2index, index2word 


def convert_to_vector_representation(data, word2index):
    vectorized_data = []

    for elt in data:
        vector = torch.zeros(len(word2index)*3) 
        for sentence in elt[1]:
            for word in sentence:
                index = word2index[word] if word in word2index else word2index[unk]
                vector[index] += 1

        for word in elt[2][0]:
            index = word2index[word] if word in word2index else word2index[unk]
            vector[index + len(word2index)] += 1

        for word in elt[2][1]:
            index = word2index[word] if word in word2index else word2index[unk]
            vector[index + 2*len(word2index)] += 1

        vectorized_data.append((elt[0], vector, elt[3]))
        print(vectorized_data[0])
    return vectorized_data


def main(hidden_dim, number_of_epochs):
    print("Fetching data")
    train_data = loader.load_data('data/train.csv')
    valid_data = loader.load_data('data/dev.csv')
    vocab = make_vocab(train_data)
    word2index, index2word = make_indices(vocab)
    print("Fetched and indexed data")

    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    print("Vectorized data")

    model = FFNN(input_dim = 3*len(vocab), h = hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("Training for {} epochs".format(number_of_epochs))
    for epoch in range(number_of_epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                _, input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                gold_label = int(gold_label) - 1
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        # loss = None
        is_correct = []
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data) # Good practice to shuffle order of validation data
        minibatch_size = 16 
        N = len(valid_data) 
        model.eval()
        for _, input_vector, gold_label in valid_data:
            gold_label = int(gold_label) - 1
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
            is_correct.append((predicted_label == gold_label, predicted_label, gold_label))
        
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        

