from __future__ import annotations
from math import log
from typing import List
import pandas as pd


def get_dataset_entropy(df: pd.DataFrame, target_attr: Attribute):
    if len(df) == 0:
        return 0

    pp = len(df[df[target_attr.label] == '1']) / len(df)
    pn = len(df[df[target_attr.label] == '0']) / len(df)

    # print(res)
    return - (pp * log(pp, 2) if pp != 0 else 0) - (pn * log(pn, 2) if pn != 0 else 0)


class Attribute:

    def __init__(self, label: str, values: List[str]):
        self.label = label
        self.values = values

    def get_gain(self, df: pd.DataFrame, target_attr: Attribute) -> float:
        target_gain = get_dataset_entropy(df, target_attr)
        df_size = len(df)

        aux = 0
        for value in self.values:
            attr_df = df[df[self.label] == value]
            aux += get_dataset_entropy(attr_df, target_attr)*len(attr_df)/df_size

        gain = target_gain - aux

        return gain
