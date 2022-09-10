from __future__ import annotations
from math import log
from typing import List
import pandas as pd

class Attribute:

    def __init__(self, label: str, values: List[str]):
        self.label = label
        self.values = values

    def get_dataset_entropy(self, df: pd.DataFrame, targetAttr: Attribute):
        pp = len(df[df[targetAttr.label] == 'P']) / len(df)
        pn = len(df[df[targetAttr.label] == 'N']) / len(df)

        return - (pp * log(pp, 2) if pp!=0 else 0) - (pn * log(pn, 2) if pn!=0 else 0)

    def get_gain(self, df: pd.DataFrame, targetAttr: Attribute) -> float:
        gain = self.get_dataset_entropy(df, targetAttr)

        for value in self.values:
            gain -= self.get_dataset_entropy(df[df[self.label] == value], targetAttr)

        return gain