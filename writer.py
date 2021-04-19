import csv
import torch

def write_preds(file_name, data, model):
    with open('preds/' + file_name + '.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['Id', 'Prediction'])

        for ID, vec in data:
            predicted_vector = model(torch.tensor(vec))
            predicted_label = torch.argmax(predicted_vector)
            writer.writerow([ID, predicted_label.item()+1])