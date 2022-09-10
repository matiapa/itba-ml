import pandas as pd

from attribute import Attribute
from decision_tree import DecisionTree
from node import TerminalState

# -------------------- DEFINITIONS --------------------

def srange(s, e):
    return list(map(lambda n : f'{n}', range(s, e+1)))

attributes = [
    Attribute('Account Balance', srange(1,4)),
    # Attribute('Duration of Credit (month)', []) --- Infinite values?
    Attribute('Payment Status of Previous Credit', srange(0,4)),

    Attribute('Purpose', srange(0,10)),
    # Attribute('Credit Amount', []) --- Infinite values?
    Attribute('Value Savings/Stocks', srange(1,5)),
    Attribute('Length of current employment', srange(1,5)),

    Attribute('Instalment per cent', srange(1,4)),
    Attribute('Sex & Marital Status', srange(1,4)),
    Attribute('Guarantors', srange(1,3)),
    Attribute('Duration in Current address', srange(1,4)),

    Attribute('Most valuable available asset', srange(1,4)),
    # Attribute('Age (years)', []) --- Infinite values?
    Attribute('Concurrent Credits', srange(1,3)),
    Attribute('Type of apartment', srange(1,3)),

    Attribute('No of Credits at this Bank', srange(1,4)),
    Attribute('Occupation', srange(1,4)),
    Attribute('No of dependents', srange(1,2)),
    Attribute('Telephone', srange(1,2)),

    Attribute('Foreign Worker', srange(1,2)),
]

targetAttribute = Attribute('Creditability', srange(0,1))

tree = DecisionTree()


# -------------------- TRAINING --------------------

df = pd.read_csv('../data/german_credit.csv')

for column in df.columns:
    df[column] = df[column].map(str)

tree.train(df, attributes, targetAttribute, 3)

# tree.print_tree()


# -------------------- TESTING --------------------

result1 = tree.evaluate({'Account Balance': '1', 'Payment Status of Previous Credit': '0', 'Purpose': '0'})

result2 = tree.evaluate({'Account Balance': '1', 'Payment Status of Previous Credit': '1', 'Purpose': '1'})

if result1 == TerminalState.NEGATIVE and result2 == TerminalState.POSITIVE:
    print('Test passed')
else:
    print('Test failed')