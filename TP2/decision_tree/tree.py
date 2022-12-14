from random import random
import pandas as pd
import numpy as np
from typing import List
import graphviz

from decision_tree.node import Node, TerminalNode, AttributeNode
from decision_tree.attribute import Attribute


class DecisionTree:

    def __init__(self, max_depth, min_samples):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def __build_tree(self, df: pd.DataFrame, attributes: List[Attribute], target_attr: Attribute, d: int) -> Node:
        if len(df) == 0:
            return TerminalNode(value='?')

        if len(df[target_attr.label].unique()) == 1 or d > self.max_depth or len(df) <= self.min_samples:
            return TerminalNode(value=df[target_attr.label].mode()[0])

        else:
            gains = {}
            for attribute in attributes:
                gains[attribute] = attribute.get_gain(df, target_attr)

            best_attr = max(gains, key=gains.get)

            newNode = AttributeNode(attribute=best_attr)
            remaining_attributes = list(attributes)
            remaining_attributes.remove(best_attr)

            for value in best_attr.values:
                subtree = self.__build_tree(df[df[newNode.attribute.label] == value], remaining_attributes, target_attr, d + 1)


                if type(subtree) is TerminalNode and subtree.value == '?':
                    subtree = TerminalNode(value=df[target_attr.label].mode()[0])
                newNode.children[value] = subtree

            return newNode

    def train(self, samples: pd.DataFrame, attributes: List[Attribute], target_attr: Attribute):
        self.root: Node = self.__build_tree(samples, attributes, target_attr, 0)

    # -------------------------------------------------------------

    def __evaluate(self, sample: dict, node: Node) -> str:
        if type(node) is TerminalNode:
            return node.value

        if type(node) is AttributeNode:
            value = sample[node.attribute.label]
            children = node.children[value]
            return self.__evaluate(sample, children)

    def evaluate(self, sample: dict) -> str:
        return self.__evaluate(sample, self.root)

    # -------------------------------------------------------------

    def __draw_node(self, node: Node, graph: graphviz.Digraph) -> str:
        if type(node) is TerminalNode:
            node_label = node.value
        else:
            node_label = node.attribute.label

        node_id = f'{random()}'

        graph.node(name=node_id, label=node_label)

        if type(node) is AttributeNode:

            for attr_value in node.attribute.values:
                child_node = node.children[attr_value]

                child_id = self.__draw_node(child_node, graph)

                graph.edge(node_id, child_id, label=attr_value)

        return node_id

    def draw_tree(self, filename: str = 'graph'):
        graph = graphviz.Digraph()

        self.__draw_node(self.root, graph)

        graph.render('out/' + filename, format='png', cleanup=True)

    # -------------------------------------------------------------

    def __trim_branch(self, root: Node, df: pd.DataFrame, target_attr: Attribute) -> Node:

        if type(root) is TerminalNode or len(df) == 0:
            return root

        final_parent_node = True
        for key, child in root.children.items():
            if type(child) is not TerminalNode:
                child = self.__trim_branch(child, df[df[root.attribute.label] == key], target_attr)
                root.children[key] = child
                if type(child) is not TerminalNode:
                    final_parent_node = False

        if final_parent_node:
            mode = df[target_attr.label].mode()[0]
            trim_errors = len(df[df[target_attr.label] != mode])

            tree_errors = 0
            for key, child in root.children.items():
                tree_errors += len(df[(df[root.attribute.label] == key) & (df[target_attr.label] != child.value)])

            if tree_errors < trim_errors:
                return root

            return TerminalNode(mode)

        return root

    def trim(self, df: pd.DataFrame, target_attr: Attribute):
        self.root = self.__trim_branch(self.root, df, target_attr)

    def __get_node_amount(self, node: Node) -> int:
        if type(node) is TerminalNode:
            return 1

        amount = 1
        for child in node.children.values():
            amount += self.__get_node_amount(child)

        return amount

    def get_node_amount(self) -> int:
        return self.__get_node_amount(self.root)
