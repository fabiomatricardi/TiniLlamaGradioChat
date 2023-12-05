# TiniLlamaGradioChat
Repo for a ChatBot, with Gradio, Streaming TinyLlama1.1BOpenOrca.gguf on CPU only

This is a ChatBot, with TinyLlama model, 1.1Billion parameters, distilled with Orca2 dataset.
## Instructions
### Virtual Environment
You can also find a requirements.txt file in the repo
Create a Virtual Environment and activate it
### Install dependencies
- langchain is required only if you want to use the conversation history capabilities.
- CTransformers is supported by LangChain
```
pip install ctransformers
pip install gradio
pip install langchain
```
### Download the files
Download the python file `Chat_tinyLlamaOpenOrca.py` and the images `456322.webp`  and `TinyLlama_logo.png`
Create a subfolder called `models` and download there the [GGUF model file tinyllama-1.1b-1t-openorca.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF).

Go in your terminal, and with the VENV activated run
```
python Chat_tinyLlamaOpenOrca.py
```

This is the result :-)
<img src="https://github.com/fabiomatricardi/TiniLlamaGradioChat/raw/main/tinillamaChat.gif" width=900>
