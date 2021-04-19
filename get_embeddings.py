import pickle

dic = {}

length = '50'

with open('../../../Downloads/glove/glove.6B.' + length + 'd.txt') as file:
    for word_vec in file.read().split('\n'):
        # print(word_vec)
        vec = word_vec.split(' ')
        # print(vec)
        dic[vec[0]] = vec[1:]

print('Done Loading')

mean = [0] * int(length)
count = 0
for key in dic.keys():
    count += 1
    for i in range(0, len(dic[key])):
        dic[key][i] = float(dic[key][i])
        mean[i] += dic[key][i]

for i in range(0, len(mean)):
    mean[i] = mean[i] / count

print(count)

pickle_out1 = open('mean.pickle', 'wb')
pickle.dump(mean, pickle_out1)
pickle_out1.close()


pickle_out2 = open('dict.pickle', 'wb')
pickle.dump(dic, pickle_out2)
pickle_out2.close()
