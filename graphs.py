import matplotlib.pyplot as plt
import csv
x = []
y = []
z = []

with open('neural_networks.csv', 'r') as file:
    plots = csv.reader(file, delimiter=',')
    for row in plots:
        print(row[0], row[1])
        if row[0] == "[8, 8, 8, 1]" and row[1] == "['relu', 'relu', 'relu', 'identity']":
            x.append(row[3])
            y.append(row[5])
        if row[0] == "[8, 8, 8, 1]" and row[1] == "['sigmoid', 'sigmoid', 'sigmoid', 'identity']":
            z.append(row[3])
            y.append(row[5])



y = sorted(y)

plt.plot(x,y, marker='o', label= "sigmoid")
plt.plot(z, y, marker='o', label= "sigmoid")

plt.title('Data from the CSV File for [128, 128, 128, 128, 128, 128, 128, 1]')

plt.xlabel('Epochs')
plt.ylabel('R2')

plt.show()