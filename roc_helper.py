import matplotlib.pyplot as plt

from scipy.io import arff
from dt_learn import DecisionTree, LeafNode


def get_confidence(root, record):
    if isinstance(root, LeafNode):
        pos, neg = root._split_values
        return float(pos + 1) / (pos + neg + 2)

    attr = root.split_on
    child = root.get_child(attr_val=record[attr])
    return get_confidence(child, record)


dt = DecisionTree("credit_train.arff", "credit_test.arff", 10)
dt.train()

test_data, test_meta = arff.loadarff("credit_test.arff")

num_neg = len(filter(lambda _class: _class == "-", test_data["class"]))
num_pos = len(test_data) - num_neg

data = []
for record in test_data:
    data.append((record, get_confidence(dt.root, record)))

data = sorted(data, key=lambda x: x[1], reverse=True)
tp = fp = 0
last_tp = 0
tpr_data = []
fpr_data = []

for i in range(len(data)):
    cur_class = data[i][0]["class"]
    if i > 1 and data[i][1] != data[i - 1][1] and cur_class == "-" and tp > last_tp:
        fpr = float(fp) / num_neg
        tpr = float(tp) / num_pos
        tpr_data.append(tpr)
        fpr_data.append(fpr)
        last_tp = tp

    if cur_class == "+":
        tp += 1
    else:
        fp += 1

fpr = float(fp) / num_neg
tpr = float(tp) / num_pos

tpr_data.append(tpr)
fpr_data.append(fpr)

print tpr_data
print fpr_data

fig, ax = plt.subplots()
ax.plot(fpr_data, tpr_data)

ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title="ROC Curve")

ax.grid()
fig.savefig("roc_curve.png")
plt.show()
