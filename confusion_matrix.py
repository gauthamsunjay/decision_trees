from collections import defaultdict
from tabulate import tabulate

matrix = {}
classes = set()

fp = open("predictions.txt", "r")
for prediction in fp.readlines():
    predicted, actual = list(map(int, prediction.strip().split(",")))
    if actual not in matrix:
        matrix[actual] = defaultdict(int)
    matrix[actual][predicted] += 1
    if actual not in classes:
        classes.add(actual)

data = []
classes = sorted(classes)
for _class in classes:
    temp = [_class]
    for k in classes:
        temp.append(matrix[_class][k])
    data.append(temp)

print tabulate(data, headers=classes, tablefmt="grid")
