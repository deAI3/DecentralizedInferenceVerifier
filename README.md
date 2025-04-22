# Readme

## 1. Build llama.cpp 
Follow the instruction in: https://github.com/ggml-org/llama.cpp 

## 2. Run the inference server
```
llama-server --model <model_name> --port 3415 --api_key token-abc123 
```
Add  ```--chat-template gemma``` if you run gemma

## 3. Install the requirement:

```
pip install -r requirements.txt
```

## 4. Add .env:
```
OPENAI_API_KEY=your_api_key
BASE_URL=your_openai_base_url
MODEL=your_model_name
```

## 5. Start the server:
```python
python app.py
```

## 6. Test:
```curl
curl -X 'POST' \
  'http://127.0.0.1:8000/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "role": "user",
    "content": "hello"
  }
]'
```