from __future__ import annotations
from typing import Dict
from decision_tree.attribute import Attribute


class Node:

    def __init__(self) -> None:
        self.children = None
        self.value = None
        self.attribute = None


class AttributeNode(Node):

    def __init__(self, attribute: Attribute):
        super().__init__()
        self.children: Dict[str, Node] = {}
        self.attribute = attribute


class TerminalNode(Node):

    def __init__(self, value: int):
        super().__init__()
        self.value = value
