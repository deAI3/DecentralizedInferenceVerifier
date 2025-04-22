import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("API_KEY")
    MODEL = os.getenv("MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 150))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    BASE_URL = os.getenv("BASE_URL", "https://api.together.xyz/v1")