from __future__ import annotations
from typing import List, Dict, Any, Iterator
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from rag_github.generation.llm_client import LLMClient
from rag_github.config.settings import get_settings

class GroqClient(LLMClient):
    def __init__(self):
        settings = get_settings()
        self.model = ChatGroq(
            model=settings.llm_model,
            temperature=0.1,
            api_key=settings.groq_api_key.get_secret_value() if hasattr(settings.groq_api_key, 'get_secret_value') else settings.groq_api_key
        )

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        langchain_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                langchain_msgs.append(SystemMessage(content=msg.get("content", "")))
            else:
                langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
        return langchain_msgs

    def generate(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        self.model.temperature = temperature
        langchain_msgs = self._convert_messages(messages)
        response = self.model.invoke(langchain_msgs)
        return response.content
        
    def stream(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> Iterator[str]:
        self.model.temperature = temperature
        langchain_msgs = self._convert_messages(messages)
        for chunk in self.model.stream(langchain_msgs):
            yield chunk.content
