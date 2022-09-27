import sys

import pandas as pd

from attributes import target_attr, attributes

sys.path.append("..")

from decision_tree.tree import DecisionTree
from decision_tree.utils import precision

# -------------------- DEFINITIONS --------------------

tree = DecisionTree(max_depth=3, min_samples=50)

# -------------------- DATA PARSING --------------------

df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
df = df.sample(frac=1).reset_index(drop=True)

train_set = df.iloc[0:600]
test_set = df.iloc[600:800]
trimming_set = df.iloc[800:1000]

# -------------------- TRAINING --------------------

tree.train(train_set, attributes, target_attr)

tree.draw_tree()

# -------------------- TESTING --------------------

print('No trimming')

print(f'> Train set: {precision(tree, train_set, target_attr)}%')
print(f'> Test set:  {precision(tree, test_set, target_attr)}%')

tree.trim(trimming_set, target_attr)
tree.draw_tree(filename='tree_trimmed')

print('With trimming')

print(f'> Train set: {precision(tree, train_set, target_attr)}%')
print(f'> Test set:  {precision(tree, test_set, target_attr)}%')
