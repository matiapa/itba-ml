import pandas as pd
import numpy as np
from typing import List

from decision_tree.node import Node, TerminalNode, AttributeNode, TerminalState
from decision_tree.attribute import Attribute
from decision_tree.utils import cprint


class DecisionTree:

    def __build_tree(self, df: pd.DataFrame, attributes: List[Attribute], targetAttr: Attribute, maxDepth: int, d: int) -> Node:
        if d > maxDepth:
            # cprint(f'Undiscriminated', 'r', d)
            return TerminalNode(state=TerminalState.UNDETERMINED)

        if len(df[df[targetAttr.label] == '0']) == len(df):
            # cprint(f'Discriminated: Negative', 'g', d)
            newNode = TerminalNode(state=TerminalState.NEGATIVE)
        
        elif len(df[df[targetAttr.label] == '1']) == len(df):
            # cprint(f'Discriminated: Positive', 'g', d)
            newNode = TerminalNode(state=TerminalState.POSITIVE)
        
        else:
            gains = map(lambda a : a.get_gain(df, targetAttr), attributes)
            best_attr = attributes[ np.argmax(gains) ]

            # cprint(f'Exploring attribute: {best_attr.label}', 'b', d)
            
            newNode = AttributeNode(attribute=best_attr)

            for value in best_attr.values:
                # cprint(f'Value: {value}', 'y', d)

                remainingAttributes = list(attributes)
                remainingAttributes.remove(best_attr)

                subtree = self.__build_tree(df[df[best_attr.label] == value], remainingAttributes, targetAttr, maxDepth, d+1)

                newNode.children[value] = subtree

        return newNode


    def train(self, df: pd.DataFrame, attributes: List[Attribute], targetAttr: Attribute, maxDepth: int):
        self.tree : Node = self.__build_tree(df, attributes, targetAttr, maxDepth, 0)


    def __evaluate(self, sample: dict, node: Node) -> TerminalState:
        if type(node) is TerminalNode:
            return node.state

        if type(node) is AttributeNode:
            value = sample[ node.attribute.label ]
            return self.__evaluate(sample, node.children[ value ])

    def evaluate(self, sample: dict):
        return self.__evaluate(sample, self.tree)


    def __print_tree(self, node: Node, d: int):
        if type(node) is TerminalNode:
            if node.state in [TerminalState.POSITIVE, TerminalState.NEGATIVE]:
                cprint(f'{node.state.name}', 'g', d)
            # else:
            #     cprint(f'{node.state.name}', 'r', d)

        if type(node) is AttributeNode:
            cprint(f'{node.attribute.label}', 'b', d)

            for value in node.attribute.values:
                cprint(f'= {value}', 'y', d)
                self.__print_tree(node.children[value], d+1)

    def print_tree(self):
        self.__print_tree(self.tree, 0)