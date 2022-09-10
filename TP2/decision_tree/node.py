from __future__ import annotations
from typing import Dict
from enum import Enum

from decision_tree.attribute import Attribute


class Node:

    def __init__(self) -> None:
        self.children : Dict[str, Node] = {}
        pass


class AttributeNode(Node):

    def __init__(self, attribute: Attribute):
        super().__init__()
        self.attribute = attribute


class TerminalState(Enum):
    POSITIVE = 1
    NEGATIVE = 2
    UNDETERMINED = 3


class TerminalNode(Node):

    def __init__(self, state: TerminalState):
        super().__init__()
        self.state = state