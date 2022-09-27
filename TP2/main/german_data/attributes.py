import sys
sys.path.append("..")
sys.path.append("../..")

from decision_tree.attribute import Attribute
from decision_tree.utils import srange

attributes = [
    Attribute('Account Balance', srange(1, 4)),
    Attribute('Duration of Credit (month)', srange(0, 6)),
    Attribute('Payment Status of Previous Credit', srange(0, 4)),

    Attribute('Purpose', srange(0, 10)),
    Attribute('Credit Amount', srange(0, 4)),
    Attribute('Value Savings/Stocks', srange(1, 5)),
    Attribute('Length of current employment', srange(1, 5)),

    Attribute('Instalment per cent', srange(1, 4)),
    Attribute('Sex & Marital Status', srange(1, 4)),
    Attribute('Guarantors', srange(1, 3)),
    Attribute('Duration in Current address', srange(1, 4)),

    Attribute('Most valuable available asset', srange(1, 4)),
    Attribute('Age (years)', srange(0, 4)),
    Attribute('Concurrent Credits', srange(1, 3)),
    Attribute('Type of apartment', srange(1, 3)),

    Attribute('No of Credits at this Bank', srange(1, 4)),
    Attribute('Occupation', srange(1, 4)),
    Attribute('No of dependents', srange(1, 2)),
    Attribute('Telephone', srange(1, 2)),

    Attribute('Foreign Worker', srange(1, 2)),
]

target_attr = Attribute('Creditability', srange(0, 1))