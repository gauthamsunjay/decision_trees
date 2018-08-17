import math
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.io import arff


class TreeNode(object):
    __metaclass__ = ABCMeta

    def __init__(self, split_on, split_values):
        self.split_on = split_on
        self.children = {}
        self._split_values = split_values

    @abstractmethod
    def get_child(self, **kwargs):
        pass

    @property
    def split_values(self):
        return "[%s %s]" % (self._split_values[0], self._split_values[1])


class NominalTreeNode(TreeNode):
    def __init__(self, split_on, split_values):
        super(NominalTreeNode, self).__init__(split_on, split_values)

    def get_child(self, **kwargs):
        if "attr_val" not in kwargs:
            raise Exception("Need to pass attr val to get the child node.")
        return self.children[kwargs["attr_val"]]


class NumericTreeNode(TreeNode):
    def __init__(self, split_on, split_values):
        super(NumericTreeNode, self).__init__(split_on, split_values)
        self.filter_functions = {
            True: lambda record: record[self.split_on] <= self.midpoint,
            False: lambda record: record[self.split_on] > self.midpoint
        }

    def get_child(self, **kwargs):
        if "attr_val" not in kwargs:
            raise Exception("Need to pass attr val to get the child node.")
        return self.children[kwargs["attr_val"] <= self.midpoint]


class LeafNode(TreeNode):
    def __init__(self, split_on, split_values):
        super(LeafNode, self).__init__(split_on, split_values)

    def get_child(self, **kwargs):
        return None


class DecisionTree(object):
    def __init__(self, train_file, test_file, m, train_percent=1):
        self.train_file = train_file
        self.test_file = test_file
        self.m = m
        self.train_percent = train_percent

    @staticmethod
    def __get_node(split_on, split_values, attr_type='nominal'):
        types = {
            'nominal': NominalTreeNode,
            'numeric': NumericTreeNode,
            'leaf': LeafNode
        }
        return types[attr_type](split_on, split_values)

    @staticmethod
    def parse_arff(filename):
        return arff.loadarff(filename)

    @staticmethod
    def __check_precedence(meta_data, attr_a, attr_b):
        return (meta_data.names().index(attr_a) <
                meta_data.names().index(attr_b))

    @staticmethod
    def __get_attr_type(attr, meta):
        return meta[attr][0]

    @staticmethod
    def __get_split_values(data, meta):
        classes = meta["class"][1]
        splits = []
        for _class in classes:
            splits.append(len(filter(lambda record: record["class"] == _class,
                                     data)))
        return splits

    @staticmethod
    def __get_attr_range(attr, data, meta):
        if meta[attr][1]:
            return meta[attr][1]
        else:
            records = sorted(data, key=lambda record: record[attr])
            midpoints = []
            for i in range(len(records) - 1):
                midpoints.append((records[i][attr] + records[i + 1][attr]) / 2.0)

            return midpoints

    @staticmethod
    def __get_label(data, meta, parent):
        classes = meta['class'][1]
        d = {}

        if len(data) > 0:
            for _class in classes:
                d[_class] = len(
                    filter(lambda record: record == _class, data['class'])
                )

            if d.values() and len(set(d.values())) <= 1:
                return parent.label

            return max(d.iterkeys(), key=d.get)
        else:
            return parent.label

    @staticmethod
    def __calculate_class_entropy(data, meta):
        classes = meta['class'][1]
        entropy = 0
        total_records = float(len(data))
        for _class in classes:
            tot = len(filter(lambda record: record['class'] == _class, data))
            prb = tot / total_records if total_records > 0 else 0
            entropy += -1 * (prb * math.log(prb, 2)) if prb > 0 else 0
        return entropy

    @staticmethod
    def __find_best_split(candidate_splits, meta):
        def get_index(attr):
            return -1 * meta.names().index(attr)

        best = sorted(
            candidate_splits,
             key=lambda split: (split["info_gain"], get_index(split["candidate"])),
            reverse=True
        )[0]

        return best["candidate"], best["attr_type"], best["split"]

    @staticmethod
    def __calculate_local_probs(data, subclasses):
        local_probs = 0
        total_records = float(len(data))
        for subclass in subclasses:
            tot = len(filter(
                lambda record: record['class'] == subclass, data)
            )
            prob = tot / total_records if total_records > 0 else 0
            local_probs += (-1 * prob * math.log(prob, 2)) if prob > 0 else 0

        return local_probs

    @staticmethod
    def __calculate_nominal_entropy(subtypes, candidate, data, subclasses):
        entropy = 0
        total_records = float(len(data))
        for subtype in subtypes:
            fdata = filter(lambda record: record[candidate] == subtype,
                           data)
            local_probs = DecisionTree.__calculate_local_probs(fdata,
                                                               subclasses)
            entropy += ((len(fdata) / total_records) * local_probs) if \
                total_records > 0 else 0
        return entropy

    @staticmethod
    def __calculate_numeric_entropy(midpoints, candidate, data, subclasses):
        total_records = float(len(data))
        potentials = []
        for midpoint in midpoints:
            entropy = 0
            for subtype in [True, False]:
                if subtype:
                    fdata = filter(
                        lambda record: record[candidate] <= midpoint, data
                    )
                else:
                    fdata = filter(
                        lambda record: record[candidate] > midpoint, data
                    )
                local_probs = DecisionTree.__calculate_local_probs(fdata,
                                                                   subclasses)
                entropy += ((len(fdata) / total_records) * local_probs)
            potentials.append((entropy, midpoint))

        if potentials:
            return sorted(potentials,
                         key=lambda potential: (potential[0], potential[1]))[0]
        else:
            return 0, 0

    def __should_stop(self, c_splits, data, max_info_gain):
        return len(c_splits) <= 0 or len(data) < self.m or \
               len(np.unique(data['class'])) == 1 or max_info_gain <= 0

    def __get_candidate_splits(self, data, meta):
        class_entropy = DecisionTree.__calculate_class_entropy(data, meta)
        splits = []
        subclasses = DecisionTree.__get_attr_range("class", data, meta)

        for candidate in self.candidates:
            attr_type = DecisionTree.__get_attr_type(candidate, meta)
            split = DecisionTree.__get_attr_range(candidate, data, meta)

            if attr_type == 'nominal':
                entropy = DecisionTree.__calculate_nominal_entropy(
                    split, candidate, data, subclasses
                )
            else:
                entropy, split = DecisionTree.__calculate_numeric_entropy(
                    split, candidate, data, subclasses
                )

            info_gain = class_entropy - entropy
            splits.append(
                {
                    "candidate": candidate,
                    "attr_type": attr_type,
                    "split": split,
                    "info_gain": info_gain
                }
            )

        return splits

    def __make_subtree(self, data, meta, parent=None):
        c_splits = self.__get_candidate_splits(data, meta)

        try:
            max_info_gain = max(c_split["info_gain"] for c_split in c_splits)
        except ValueError:
            max_info_gain = -1

        split_values = DecisionTree.__get_split_values(data, meta)

        if self.__should_stop(c_splits, data, max_info_gain):
            node = DecisionTree.__get_node(None, split_values,
                                           attr_type='leaf')
            node.label = self.__get_label(data, meta, parent)
        else:
            best_split, attr_type, split = self.__find_best_split(c_splits,
                                                                  meta)
            node = DecisionTree.__get_node(best_split, split_values,
                                           attr_type=attr_type)
            node.label = self.__get_label(data, meta, parent)

            if isinstance(node, NominalTreeNode):
                self.candidates.remove(best_split)
                for subtype in split:
                    filtered_data = np.array(filter(
                        lambda record: record[best_split] == subtype,
                        data
                    ))
                    node.children[subtype] = self.__make_subtree(
                        filtered_data, meta, parent=node
                    )
            else:
                node.midpoint = split
                for subtype in [True, False]:
                    filtered_data = np.array(filter(
                        node.filter_functions[subtype], data)
                    )
                    node.children[subtype] = self.__make_subtree(
                        filtered_data, meta, parent=node
                    )

        return node

    def train(self):
        train_data, train_meta = DecisionTree.parse_arff(self.train_file)

        def traverse(root, num_tabs=0):
            if num_tabs == 0:
                pt_str = ""
            else:
                pt_str = "|\t" * num_tabs

            if isinstance(root, NumericTreeNode):
                for val in [True, False]:
                    sign = "<=" if val else ">"
                    ptr = pt_str + "%s " % root.split_on.lower() + \
                          "%s" % sign + " %.6f " % root.midpoint + \
                          root.children[val].split_values

                    if isinstance(root.children[val], LeafNode):
                        print ptr + ": %s" % root.children[val].label
                    else:
                        print ptr
                        traverse(root.children[val], num_tabs+1)
            else:
                for val in train_meta[root.split_on][1]:
                    ptr = pt_str + "%s " % root.split_on.lower() + \
                                "= " + val + " " + \
                                root.children[val].split_values

                    if isinstance(root.children[val], LeafNode):
                        print ptr + ": %s" % root.children[val].label
                    else:
                        print ptr
                        traverse(root.children[val], num_tabs+1)

        self.candidates = set(train_meta.names()[:-1])
        if self.train_percent < 1:
            data_len = int(len(train_data) * self.train_percent)
            train_data = np.random.choice(train_data, data_len, replace=False)

        node = self.__make_subtree(train_data, train_meta)
        self.root = node
        traverse(self.root)

    def test(self):
        print "<Predictions for the Test Set Instances>"
        test_data, test_meta = DecisionTree.parse_arff(self.test_file)
        accurately_predicted = 0

        def get_label(root, record):
            if isinstance(root, LeafNode):
                return root.label

            attr = root.split_on
            return get_label(root.get_child(attr_val=record[attr]), record)

        for i, record in enumerate(test_data, 1):
            actual = record["class"]
            predicted = get_label(self.root, record)
            print "%d: Actual: %s Predicted: %s" % (i, actual, predicted)
            if predicted == actual:
                accurately_predicted += 1

        print "Number of correctly classified: %d " \
              "Total number of test instances: %d" % (accurately_predicted,
                                                      len(test_data))

        return (float(accurately_predicted) / len(test_data)) * 100


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print "python dt-learn.py <train-set-file> <test-set-file> m"
        exit(1)

    train_file, test_file, m = sys.argv[1:]
    dt = DecisionTree(train_file, test_file, int(m))
    dt.train()
    dt.test()
