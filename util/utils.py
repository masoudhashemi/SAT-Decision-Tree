
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.tree import _tree


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
    
def get_lineage(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for child in idx:
        for node in recurse(left, right, child):
            print(node)
            
            
def get_thresholds(dt, feature_names):
    tree = dt.tree_
    dict_feat_thresh = {}
    for i in range(len(tree.feature)):
        if tree.feature[i]!=_tree.TREE_UNDEFINED:
            if dict_feat_thresh.get(feature_names[tree.feature[i]], None) is None:
                dict_feat_thresh[feature_names[tree.feature[i]]] = [tree.threshold[i]]
            else:
                dict_feat_thresh[feature_names[tree.feature[i]]].append(tree.threshold[i])
    for k,v in dict_feat_thresh.items():
        dict_feat_thresh[k] = list(set(v))
    return dict_feat_thresh
