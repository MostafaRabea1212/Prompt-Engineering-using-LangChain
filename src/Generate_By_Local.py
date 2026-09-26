from transformers import AutoTokenizer, AutoModelForCausalLM ,BitsAndBytesConfig ,pipeline
import torch
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

class BaseLLMLocal:

    def __init__(self , prompt : str):
        self.prompt=prompt
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model_id = "Qwen/Qwen3-4B-Instruct-2507"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=self.quantization_config
        )

        self.hf_pipeline = pipeline('text-generation',
                        model=self.base_model,
                       tokenizer=self.tokenizer,
                       max_length=256)
        
        self.llm=HuggingFacePipeline(pipeline=self.hf_pipeline)
        
        self.messages = [
                {"role": "user", "content": self.prompt}
            ]

        self.text =self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
    def invoke(self):
        response=self.llm.invoke(self.text)
        return response