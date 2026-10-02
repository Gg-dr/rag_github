from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator

class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        pass
        
    @abstractmethod
    def stream(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> Iterator[str]:
        pass
