from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


class BaseLLM:

    def __init__(self):
        self.llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0,
            max_tokens=400,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )

    def invoke(self, prompt):
        response = self.llm.invoke(prompt)
        return response

    def generate(self, prompts):
        response = self.llm.generate(prompts)
        return response