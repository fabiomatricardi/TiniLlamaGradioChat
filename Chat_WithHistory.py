import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGUF/GGML models
import datetime

i_temperature = 0.30 
i_max_new_tokens=1100
modelfile = "models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
i_repetitionpenalty = 1.2
i_contextlength=12048
logfile = 'TinyLlamaOpenOrca1.1B-stream.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=i_temperature, repetition_penalty=i_repetitionpenalty, batch_size=64,
                max_new_tokens=i_max_new_tokens, context_length=i_contextlength))
llm = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama",config = conf)
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")
#MODEL SETTINGS also for DISPLAY

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

with gr.Blocks(theme='ParityError/Interstellar') as demo: 
    #TITLE SECTION
    with gr.Row():
        with gr.Column(scale=12):
            gr.HTML("<center>"
            + "<h1>🦙 TinyLlama 1.1B 🐋 OpenOrca 4K context window</h2></center>")  
            gr.Markdown("""
            **Currently Running**: [tinyllama-1.1b-1t-openorca.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Chat History Log File**: *TinyLlamaOpenOrca1.1B-stream.txt*  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **License**: Apache 2.0, following the TinyLlama base model.
            - **Base Model**: PY007/TinyLlama-1.1B-intermediate-step-480k-1T,  Fine tuned on OpenOrca GPT4 subset for 1 epoch,Using CHATML format. The model output is not censored. Use at your own risk.""")         
        gr.Image(value='./TinyLlama_logo.png', height=90)
   # chat and parameters settings
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height = 425, show_copy_button=True,
                                 avatar_images = ["./456322.webp","./TinyLlama_logo.png"])
            with gr.Row():
                with gr.Column(scale=14):
                    msg = gr.Textbox(show_label=False, 
                                     placeholder="Enter text",
                                     lines=2)
                submitBtn = gr.Button("\n💬 Send\n", size="lg", variant="primary", min_width=180)

        with gr.Column(min_width=100,scale=2):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.30,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=360,
                        step=4,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    rep_pen = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1.2,
                        step=0.05,
                        interactive=True,
                        label="Repetition Penalty",
                    )
                    gr.Markdown("""
                    ### History lenght
                    Insert num of chat rounds for conversation context
                    """)
                    mem_limit = gr.Slider(
                        minimum=5,
                        maximum=12,
                        value=8,
                        step=1,
                        interactive=True,
                        label="Chat History Lenght",
                    )

                clear = gr.Button("🗑️ Clear All Messages", variant='secondary')
    def user(user_message, history):
        writehistory(f"USER: {user_message}")
        return "", history + [[user_message, None]]

    def bot(history,t,m,r,limit):
        mhistory = []
        SYSTEM_PROMPT = """<|im_start|>system
        You are a helpful bot. Your answers are clear and concise.
        <|im_end|>

        """
        # always keep len(history) <= memory_limit
        if len(history) > limit:
            # using mhistory instead of history to avoid that the chat interface is also truncated
            # we want only the CONTEXT for the chat to be limited to the number of rounds
            mhistory = history[-limit:]   
            print("History above set limit")
        else:
            mhistory = history
        # First prompt different because does not contain any context            
        if len(history) == 1:
            prompt = f"<|im_start|>system<|im_end|><|im_start|>user\n{history[-1][0]}<|im_end|>\n<|im_start|>assistant\n"  
            print(f"History = 1. history lenght: {len(history)}")
        # Prompt woth context
        else:
            print(f"History > 1. history lenght: {len(history)}")
            intro = f'''<|im_start|>system<|im_end|><|im_start|>user\nReply to the user honestly. 
            If you cannot find the answer in the given context, reply "Here what I can find about your question" 
            and use your own knowledge to reply.[Context]\n'''
            context = ""
            for item in mhistory:
                context = context + f"user: {item[0]}\nassistant: {item[0]}\n"

            context = context + "[end of Context]\n"
            prompt = intro + context + f"user\n{history[-1][0]}<|im_end|>\n<|im_start|>assistant\n"
        # Preparing the CHATBOT reply
        history[-1][1] = ""
        for character in llm(prompt, 
                 temperature = t,
                 repetition_penalty = r, 
                 max_new_tokens=m,
                 stop = ['<|im_end|>'],
                 stream = True):
            history[-1][1] += character
            yield history
        writehistory(f"temperature: {t}, maxNewTokens: {m}, repetitionPenalty: {r}\n---\nBOT: {history}\n\n")
        #Log in the terminal the messages
        print(f"USER: {history[-1][0]}\n---\ntemperature: {t}, maxNewTokens: {m}, repetitionPenalty: {r}\n---\nBOT: {history[-1][1]}\n\n")    
    # Clicking the submitBtn will call the generation with Parameters in the slides
    submitBtn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot,temperature,max_length_tokens,rep_pen,mem_limit], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()  #required to yield the streams from the text generation
demo.launch(inbrowser=True)
