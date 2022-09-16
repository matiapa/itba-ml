from random import random
import pandas as pd
import numpy as np
from typing import List
import graphviz

from node import Node, TerminalNode, AttributeNode
from attribute import Attribute

class DecisionTree:

    def __init__(self, maxDepth, minSamples):
        self.maxDepth = maxDepth
        self.minSamples = minSamples


    def __build_tree(self, df: pd.DataFrame, attributes: List[Attribute], targetAttr: Attribute, d: int) -> Node:
        if len(df) == 0:
            return TerminalNode(value = '?')

        if len(df[targetAttr.label].unique()) == 1 or d > self.maxDepth or len(df) <= self.minSamples:
            return TerminalNode(value = df[targetAttr.label].mode()[0])
        
        else:
            gains = map(lambda a : a.get_gain(df, targetAttr), attributes)
            best_attr = attributes[ np.argmax(gains) ]
            
            newNode = AttributeNode(attribute=best_attr)

            for value in best_attr.values:
                remainingAttributes = list(attributes)
                remainingAttributes.remove(best_attr)

                subtree = self.__build_tree(df[df[best_attr.label] == value], remainingAttributes, targetAttr, d+1)

                newNode.children[value] = subtree

            return newNode


    def train(self, samples: pd.DataFrame, attributes: List[Attribute], targetAttr: Attribute):
        self.root : Node = self.__build_tree(samples, attributes, targetAttr, 0)

    # -------------------------------------------------------------

    def __evaluate(self, sample: dict, node: Node) -> str:
        if type(node) is TerminalNode:
            return node.value

        if type(node) is AttributeNode:
            value = sample[ node.attribute.label ]
            return self.__evaluate(sample, node.children[ value ])

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

    def draw_tree(self):
        graph = graphviz.Digraph()

        self.__draw_node(self.root, graph)

        graph.render('out/graph', format='png', cleanup=True)


    # -------------------------------------------------------------


    def __trim_branch(self, root: Node, df: pd.DataFrame, targetAttr: Attribute) -> Node:
        if type(root) is TerminalNode:
            return root

        temp_tree = DecisionTree(None, None)
        temp_tree.root = root
        untrimmed_corrects = 0

        for _, sample in df.iterrows():
            sample = sample.to_dict()
            prediction = temp_tree.evaluate(sample)

            if prediction == sample[targetAttr.label]:
                untrimmed_corrects += 1

        mode = df[targetAttr.label].mode()[0]
        trimmed_corrects = len(df[df[targetAttr.label] == mode])

        if trimmed_corrects > untrimmed_corrects:
            return TerminalNode(value = mode)

        else:
            new_children = []

            for attr_value, child in root.children:
                new_child = self.__trim_branch(child, df[df[targetAttr.label] == attr_value])
                new_children.append(new_child)
            root.children = new_children

            return root

    def trim(self, df: pd.DataFrame, targetAttr: Attribute):
        self.root = self.__trim_branch(self, df, targetAttr)