from random import randint
import matplotlib.pyplot as plt
import csv
x = []
y = []
z = []
q = []

def parseString(string):
    string = string[1:-1]
    return string.split(',')[0]



with open('neural_networks.csv', 'r') as file:
    plots = csv.reader(file, delimiter=',')
    for row in plots:
        if "relu" in row[1]:
            x.append(row[5])
            parseString(row[0])
            q.append(parseString(row[0]))
        elif "sigmoid" in row[1]:
            z.append(row[5])
            y.append(parseString(row[0]))


y = list(map(lambda x: round(float(x),4),y))
q = list(map(lambda x: round(float(x),4),q))
z = list(map(lambda x: round(float(x),4),z))
x = list(map(lambda x: round(float(x),4),x))

plt.scatter(y,z, marker='o', label= "sigmoid")
plt.scatter(q,x, marker='x', label= "relu")

plt.title('Data from the CSV File')

plt.ylabel('R2')
plt.xlabel('Neurons per Layers')
plt.legend()

plt.show()