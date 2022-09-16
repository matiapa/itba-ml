from __future__ import annotations
from typing import Dict
from enum import Enum

from attribute import Attribute


class Node:

    def __init__(self) -> None:
        pass


class AttributeNode(Node):

    def __init__(self, attribute: Attribute):
        super().__init__()
        self.children : Dict[str, Node] = {}
        self.attribute = attribute


class TerminalNode(Node):

    def __init__(self, value: str):
        super().__init__()
        self.value = value