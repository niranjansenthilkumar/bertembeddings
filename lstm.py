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
import pickle
import loader
import random
import writer

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.U = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 2)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()

        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

        #self.network = nn.RNN(input_size = input_size, hidden_size = hidden_size, nonlinearity = 'tanh')
        self.network = nn.LSTM(input_size = input_size, hidden_size = hidden_size)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)  

    def forward(self, inputs):
        # hidden = torch.zeros([1, self.hidden_size])
        # output = []
        # for i in inputs:
        #     hidden = self.activation2(self.U(hidden) + self.W(i))
        #     y = self.activation2(self.V(hidden))
        #     output.append(y)

        # #print(output)
        # predicted_vector = self.softmax(output[-1])
        # #print(predicted_vector)

        # return predicted_vector[0]
        #print('start')
        #print(inputs)
        output_1 = self.network(inputs.unsqueeze(1))[0]
        #print(output_1)
        output_2 = self.activation2(self.V(output_1))
        #print(output_2)
        last_vec = output_2[-1][0]
        #print(last_vec)
        #print('end')
        predicted_vector = self.softmax(last_vec)
        return predicted_vector


def convert_to_vector_representation(data, embeddings, mean, test):
    END = [1000] * 50
    vectorized_data = []
    #print(data[:1])
    for elt in data:
        story = []
        for sentence in elt[1]:
            lst = sentence[:len(sentence)-1].lower().split(' ') + [sentence[-1]]
            for word in lst:
                embed = embeddings[word] if word in embeddings else mean
                if embed != []:
                    #vector.append(torch.tensor(embed, dtype=torch.float))
                    story.append(embed)

        #vector.append(torch.tensor(END, dtype=torch.float))
        story.append(END)

        vector1 = []

        lst = elt[2][0][:len(elt[2][0])-1].lower().split(' ') + [elt[2][0][-1]]
        for word in lst:
            embed = embeddings[word] if word in embeddings else mean
            if embed != []:
                #vector.append(torch.tensor(embed, dtype=torch.float))
                vector1.append(embed)

        #vector.append(torch.tensor(END, dtype=torch.float))
        vector1.append(END)
        
        vector1 = story + vector1

        #vector2 = []
        
        lst = elt[2][1][:len(elt[2][1])-1].lower().split(' ') + [elt[2][1][-1]]
        for word in lst:
            embed = embeddings[word] if word in embeddings else mean
            if embed != []:
                #vector.append(torch.tensor(embed, dtype=torch.float))
                vector1.append(embed)

        #vector.append(torch.tensor(END, dtype=torch.float))
        vector1.append(END)

        #vector2 = story + vector2

        #print(elt[3])
        #vector1 = [[float(elt[3]==0)]]
        #vector2 = [[float(elt[3]==1)]]

        #vector1 = [[float(elt[3])]]
        if not(test):
            vectorized_data.append((elt[0], vector1, elt[3]))
        else:
            vectorized_data.append((elt[0], vector1))

    return vectorized_data


def main(hidden_dim, number_of_epochs):
    print('RNN running')
    print("Fetching data")
    train_data = loader.load_data('data/train.csv', False)
    valid_data = loader.load_data('data/dev.csv', False)
    test_data = loader.load_data('data/test.csv', True)
    #vocab = make_vocab(train_data)
    print("Fetched and indexed data")

    pickle_in1 = open("dict.pickle", "rb")
    wordEmbeddings = pickle.load(pickle_in1)

    pickle_in2 = open("mean.pickle", "rb")
    meanVec = pickle.load(pickle_in2)
    
    train_data = convert_to_vector_representation(train_data, wordEmbeddings, meanVec, False)
    valid_data = convert_to_vector_representation(valid_data, wordEmbeddings, meanVec, False)
    test_data = convert_to_vector_representation(test_data, wordEmbeddings, meanVec, True)
    #train_data = train_data[:32]
    #train_data[0] = (train_data[0][0], train_data[0][1], 0)
    #valid_data = train_data
    print("Vectorized data")

    model = RNN(hidden_size = hidden_dim, input_size = 50)
    #model = nn.LSTM(50, 2)
    optimizer = optim.SGD(model.parameters(),lr=.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters())
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
        totalloss = None
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                _, input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]

                predicted_vector = model(torch.tensor(input_vector))

                #inputs = torch.cat(input_vector).view(len(input_vector), 1, -1)
                #hidden = (torch.randn(1, 1, 2), torch.randn(1, 1, 2))
                #predicted_vector, _ = model(torch.tensor(inputs), hidden)
                #pred1, pred2 = model(torch.tensor(input_vectors[0])), model(torch.tensor(input_vectors[1]))


                #print('new epoch')
                #print(pred1)
                #print(pred2)


                #predicted_label1, predicted_label2 = torch.argmax(pred1), torch.argmax(pred2)

                predicted_label = torch.argmax(predicted_vector)

                # predicted_label = None
                # if predicted_label1 == 1 and predicted_label2 == 0:
                #     predicted_label = 0
                # elif predicted_label1 == 0 and predicted_label2 == 1:
                #     predicted_label = 1
                # else:
                #     predicted_label = random.randint(0,1)

                correct += int(predicted_label == gold_label)
                total += 1
                #example_loss1 = model.compute_Loss(pred1.view(1,-1), torch.tensor([int(gold_label == 0)]))
                #example_loss2 = model.compute_Loss(pred2.view(1,-1), torch.tensor([int(gold_label == 1)]))
                #example_loss = example_loss1 + example_loss2

                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))

                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size

            if totalloss is None:
                totalloss = loss
            else:
                totalloss += loss

            loss.backward()
            optimizer.step()

        print(totalloss)
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
            #inputs = torch.cat(input_vector).view(len(input_vector), 1, -1)
            #hidden = (torch.randn(1, 1, 2), torch.randn(1, 1, 2))
            #predicted_vector, _ = model(torch.tensor(inputs), hidden)

            #pred1, pred2 = model(torch.tensor(input_vectors[0])), model(torch.tensor(input_vectors[1]))

            #predicted_label1, predicted_label2 = torch.argmax(pred1), torch.argmax(pred2)

            #predicted_label = torch.argmax(torch.tensor([pred1[1], pred2[1]]))

            predicted_vector = model(torch.tensor(input_vector))

            predicted_label = torch.argmax(predicted_vector)

            # predicted_label = None
            # if predicted_label1 == 1 and predicted_label2 == 0:
            #     predicted_label = 0
            # elif predicted_label1 == 0 and predicted_label2 == 1:
            #     predicted_label = 1
            # else:
            #     predicted_label = random.randint(0,1)

            #predicted_vector = model(torch.tensor(input_vector))
            #predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
            is_correct.append((predicted_label == gold_label, predicted_label, gold_label))
        
        print(is_correct)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        if epoch >= 15:
            writer.write_preds('test_preds' + str(epoch),test_data,model)
        
