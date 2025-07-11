from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from config import CasualLMConfig
from typing import List, Dict, Optional, Any
import torch
import pickle
import base64
import json
import random
import uvicorn

app = FastAPI()
model = AutoModelForCausalLM.from_pretrained(CasualLMConfig.MODEL_NAME, trust_remote_code=True, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(CasualLMConfig.MODEL_NAME, trust_remote_code=True)

class ChatResponse:
    output: str
    stop: bool = False
    kv_cache: Optional[str] = None
@app.get("/")
async def root():
    return {"message": "Welcome to the OpenAI Streaming Chat API"}

@app.post("/chat/verify")
def verify(messages: List[dict], kv_cache, max_tokens: Optional[int], generator_results: List[str]):
    try:
        prompt = ""
        for message in messages:
            if message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Deserialize KV cache if provided
        past_key_values = deserialize_kv_cache(kv_cache)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                past_key_values=past_key_values,
                max_new_tokens=max_tokens,
                temperature=CasualLMConfig.TEMPERATURE,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
    
        new_tokens = outputs.sequences[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        votes_indices = verify_step(generated_text, generator_results)
        
        return votes_indices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/completions")
def chat(messages: List[dict], kv_cache, max_tokens: Optional[int]):
    try:
        prompt = ""
        for message in messages:
            if message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        
        # Deserialize KV cache if provided
        past_key_values = deserialize_kv_cache(kv_cache)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                past_key_values=past_key_values,
                max_new_tokens=max_tokens,
                temperature=CasualLMConfig.TEMPERATURE,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
    
        new_tokens = outputs.sequences[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Serialize new KV cache
        new_kv_cache = serialize_kv_cache(outputs.past_key_values)

        stop = False
        if generated_text.endswith(tokenizer.eos_token or "</s>"):
            stop = True
        response = ChatResponse(
            output=generated_text,
            stop=stop,
            kv_cache=new_kv_cache,
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def verify_step(generated_text: str, generator_results: List[str]) -> List[int]:
    """
    Verify the generated text against the expected results.
    Returns a list of indices where the generated text matches the expected results.
    """
    verified_indices = []
    for i, expected in enumerate(generator_results):
        if generated_text.strip() == expected.strip():
            verified_indices.append(i)
    return verified_indices
    

def serialize_kv_cache(past_key_values) -> str:
    """Serialize KV cache to base64 string"""
    if past_key_values is None:
        return None
    
    # Convert tensors to CPU and serialize
    cpu_cache = []
    for layer_cache in past_key_values:
        layer_cpu = []
        for tensor in layer_cache:
            layer_cpu.append(tensor.cpu())
        cpu_cache.append(tuple(layer_cpu))
    
    serialized = pickle.dumps(tuple(cpu_cache))
    return base64.b64encode(serialized).decode('utf-8')

def deserialize_kv_cache(cache_str: str):
    """Deserialize KV cache from base64 string"""
    if not cache_str:
        return None
    
    try:
        serialized = base64.b64decode(cache_str.encode('utf-8'))
        cpu_cache = pickle.loads(serialized)
        
        # Move tensors back to GPU if available
        device = next(model.parameters()).device
        gpu_cache = []
        for layer_cache in cpu_cache:
            layer_gpu = []
            for tensor in layer_cache:
                layer_gpu.append(tensor.to(device))
            gpu_cache.append(tuple(layer_gpu))
        
        return tuple(gpu_cache)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid KV cache: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run("app:app")