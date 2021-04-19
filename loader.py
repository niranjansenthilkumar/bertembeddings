import csv

def load_data(path, test):
	lst = []

	with open(path) as file:
		reader = csv.reader(file, delimiter=',')
		first = True
		for row in reader:
			if not(first):
				ID = row[0]
				story = [row[1],row[2],row[3],row[4]]
				endings = [row[5],row[6]]
				if not(test):
					label = int(row[7]) - 1

					lst.append([ID, story, endings, label])
				else:
					lst.append([ID, story, endings])
				
			first = False

	return lst