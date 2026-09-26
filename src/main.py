from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGroq(
    model="qwen/qwen3.8-27b",  
    temperature=0,
    max_tokens=1024,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    )

prompt= """
Suggest 2 ways to lose my weight.
""".strip()

response=llm.invoke(prompt)
print(response.content)
