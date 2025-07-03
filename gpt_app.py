from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from config import GPTConfig
from typing import List, Dict, Optional, Any
import json
import uvicorn


app = FastAPI()

client = OpenAI(
    base_url=GPTConfig.BASE_URL,
    api_key=GPTConfig.OPENAI_API_KEY,
)

verify_client = OpenAI(
    base_url=GPTConfig.BASE_URL,
    api_key=GPTConfig.OPENAI_API_KEY,
)

@app.get("/")
async def root():
    return {"message": "Welcome to the OpenAI Streaming Chat API"}


tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")

def verify_step(messages: List[dict], current_response: str, next_token: str):
    cur_messages = messages.copy()
    cur_messages.append({"role": "assistant", "content": current_response})
    prompt = tokenizer.apply_chat_template(
        cur_messages,
        tokenize=False,
        add_generation_prompt=False,
    )[:-len("<end_of_turn>")-1]
    next_predict_token = verify_client.completions.create(
        model=GPTConfig.MODEL,
        prompt=prompt,
        stream=False,
        top_p=0.95,
        max_tokens=1,
    )
    print("DEBUG", next_predict_token)
    next_predict_token = next_predict_token.choices[0].text
    if next_predict_token != next_token:
        print("DEBUG", prompt, "\n", next_predict_token, next_token)
        return False
    return True

@app.post("/chat/completions")
def chat(messages: List[dict]):
    def generate(messages: List[dict]):
        try:
            stream = client.chat.completions.create(
                model=GPTConfig.MODEL,
                messages=messages,
                stream = True,
                # temperature=0
                top_p=0.95,
            )
            current_response = ""
            for chunk in stream:
                current_token = chunk.choices[0].delta.content
                if random.randint(0, 100) < 30 and current_token not in ["\n", " ", "*", ""]:
                    if verify_step(messages.copy(), current_response, current_token):
                        print("DEBUG", "CORRECT")
                    else:
                        print("DEBUG", "INCORRECT")
                if current_token != None:
                    current_response += current_token
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            print("ERROR", e)
    
    return StreamingResponse(generate(messages),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}
        )

if __name__ == "__main__":
    uvicorn.run("app:app")