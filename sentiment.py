from pycorenlp import StanfordCoreNLP
import loader
import pickle


train_data = loader.load_data('data/train.csv', False)
valid_data = loader.load_data('data/dev.csv', False)
test_data = loader.load_data('data/test.csv', True)

nlp = StanfordCoreNLP('http://localhost:9000')

features = []

for elt in train_data:
	print(' '.join(elt[1]))
	print(elt[2][0])
	print(elt[2][1])
	feature_vec = []
	story = nlp.annotate(' '.join(elt[1]),
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	end1 = nlp.annotate(elt[2][0],
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	end2 = nlp.annotate(elt[2][1],
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	feature_vec.append([])
	for s in story['sentences']:
		feature_vec[0].append(s['sentimentValue'])

	feature_vec.append(end1['sentences'][0]['sentimentValue'])
	feature_vec.append(end2['sentences'][0]['sentimentValue'])
	
	print(feature_vec)

	features.append(feature_vec)
	if len(feature_vec[0]) != len(elt[1]):
		print('uh oh')

pickle_out_train = open('train_sentiment.pickle', 'wb')
pickle.dump(features, pickle_out_train)
pickle_out_train.close()


features = []

for elt in valid_data:
	feature_vec = []
	story = nlp.annotate(' '.join(elt[1]),
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	end1 = nlp.annotate(elt[2][0],
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	end2 = nlp.annotate(elt[2][1],
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	feature_vec.append([])
	for s in story['sentences']:
		feature_vec[0].append(s['sentimentValue'])

	feature_vec.append(end1['sentences'][0]['sentimentValue'])
	feature_vec.append(end2['sentences'][0]['sentimentValue'])
	
	features.append(feature_vec)
	if len(feature_vec[0]) != len(elt[1]):
		print('uh oh')

pickle_out_valid = open('valid_sentiment.pickle', 'wb')
pickle.dump(features, pickle_out_valid)
pickle_out_valid.close()

features = []

for elt in test_data:
	feature_vec = []
	story = nlp.annotate(' '.join(elt[1]),
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	end1 = nlp.annotate(elt[2][0],
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	end2 = nlp.annotate(elt[2][1],
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })

	feature_vec.append([])
	for s in story['sentences']:
		feature_vec[0].append(s['sentimentValue'])

	feature_vec.append(end1['sentences'][0]['sentimentValue'])
	feature_vec.append(end2['sentences'][0]['sentimentValue'])
	
	features.append(feature_vec)
	if len(feature_vec[0]) != len(elt[1]):
		print('uh oh')

pickle_out_test = open('test_sentiment.pickle', 'wb')
pickle.dump(features, pickle_out_test)
pickle_out_test.close()