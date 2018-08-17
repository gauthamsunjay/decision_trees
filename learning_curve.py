import sys
from collections import defaultdict

from dt_learn import DecisionTree

if len(sys.argv) < 4:
    print "Usage: python learning_curve.py <train_file> <test_file> m"

data = defaultdict(list)

train_file, test_file, m = sys.argv[1:]

for tp in [0.05, 0.1, 0.2, 0.5, 1]:
    if tp == 1:
        dt = DecisionTree(train_file, test_file, int(m), float(tp))
        dt.train()
        data[tp].append(dt.test())
        continue

    for _ in range(10):
        dt = DecisionTree(train_file, test_file, int(m), float(tp))
        dt.train()
        data[tp].append(dt.test())

for tp in [0.05, 0.1, 0.2, 0.5, 1]:
    print "tp = %f" % tp
    print "Min = %f, Avg = %f, Max = %f" % (min(data[tp]),
                                            (float(sum(data[tp])) / len(data[tp])),
                                            max(data[tp]))
    print "=" * 80




