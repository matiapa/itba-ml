import pandas as pd

from attribute import Attribute
from tree import DecisionTree

# -------------------- DEFINITIONS --------------------

def srange(s, e):
    return list(map(lambda n : f'{n}', range(s, e+1)))

def precision(tree, df, targetAttr):
    return round(sum([1 if tree.evaluate(s) == s[targetAttr.label] else 0 for _,s in df.iterrows()]) / len(df) * 100)

attributes = [
    Attribute('Account Balance', srange(1,4)),
    Attribute('Duration of Credit (month)', srange(0,3)),
    Attribute('Payment Status of Previous Credit', srange(0,4)),

    Attribute('Purpose', srange(0,10)),
    Attribute('Credit Amount', srange(0,3)),
    Attribute('Value Savings/Stocks', srange(1,5)),
    Attribute('Length of current employment', srange(1,5)),

    Attribute('Instalment per cent', srange(1,4)),
    Attribute('Sex & Marital Status', srange(1,4)),
    Attribute('Guarantors', srange(1,3)),
    Attribute('Duration in Current address', srange(1,4)),

    Attribute('Most valuable available asset', srange(1,4)),
    Attribute('Age (years)', srange(0,3)),
    Attribute('Concurrent Credits', srange(1,3)),
    Attribute('Type of apartment', srange(1,3)),

    Attribute('No of Credits at this Bank', srange(1,4)),
    Attribute('Occupation', srange(1,4)),
    Attribute('No of dependents', srange(1,2)),
    Attribute('Telephone', srange(1,2)),

    Attribute('Foreign Worker', srange(1,2)),
]

targetAttr = Attribute('Creditability', srange(0,1))

tree = DecisionTree(maxDepth = 3, minSamples = 10)


# -------------------- DATA PARSING --------------------

df = pd.read_csv('../data/german_credit_proc.csv')

for column in df.columns:
    df[column] = df[column].map(str)

df = df.sample(frac=1).reset_index(drop=True)

trainSet = df.iloc[0:900]
testSet = df.iloc[900:1000]


# -------------------- TRAINING --------------------

tree.train(trainSet, attributes, targetAttr)

tree.draw_tree()


# -------------------- TESTING --------------------

print('No trimming')

print(f'> Train set: {precision(tree, trainSet, targetAttr)}%')
print(f'> Test set:  {precision(tree, testSet, targetAttr)}%')

tree.trim(testSet, targetAttr)

print('With trimming')

print(f'> Train set: {precision(tree, trainSet, targetAttr)}%')
print(f'> Test set:  {precision(tree, testSet, targetAttr)}%')