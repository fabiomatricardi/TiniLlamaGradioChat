import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime

i_temperature = 0.32 
i_max_new_tokens=1100
modelfile = "models/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
i_repetitionpenalty = 1.15
i_contextlength=12048
logfile = 'TinyLlamaOpenOrca1.1B-stream.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=i_temperature, repetition_penalty=i_repetitionpenalty, batch_size=64,
                max_new_tokens=i_max_new_tokens, context_length=i_contextlength))
llm = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama",config = conf) #model_type="stablelm", 
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")
#MODEL SETTINGS also for DISPLAY

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

with gr.Blocks(theme='ParityError/Interstellar') as demo: #theme='ParityError/LimeFace'
    #TITLE SECTION
    with gr.Row():
        with gr.Column(scale=12):
            gr.HTML("<center>"
            + "<h1>🦙 TinyLlama 1.1B 🐋 OpenOrca 4K context window</h2></center>")  
            gr.Markdown("""
            Currently Running: [tinyllama-1.1b-1t-openorca.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF)

            - **Base Model**: PY007/TinyLlama-1.1B-intermediate-step-480k-1T,  Fine tuned on OpenOrca GPT4 subset for 1 epoch,Using CHATML format. 
            - **License**: Apache 2.0, following the TinyLlama base model. The model output is not censored and the authors do not endorse the opinions in the generated content. Use at your own risk.
            """)         
        #with gr.Column(scale=1):
        gr.Image(value='./TinyLlama_logo.png', width=70)
   # chat and parameters settings
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height = 350, show_copy_button=True,
                                 avatar_images = ["./456322.webp","./TinyLlama_logo.png"])
            with gr.Row():
                with gr.Column(scale=14):
                    msg = gr.Textbox(show_label=False, 
                                     placeholder="Enter text",
                                     lines=2)#
                #with gr.Column(min_width=5, ): #scale=1
                #    pass
                submitBtn = gr.Button("\n💬 Send\n", size="lg", variant="primary", min_width=180)
                #with gr.Column(min_width=100, scale=2):
                #    clear = gr.Button("\n🗑️ Clear\n", variant='secondary')                                  
            #clear = gr.Button("🗑️ Clear All Messages", variant='secondary') #moved below parameters
        with gr.Column(min_width=50,scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.32,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=1200,
                        step=4,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    rep_pen = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1.15,
                        step=0.05,
                        interactive=True,
                        label="Repetition Penalty",
                    )
                clear = gr.Button("🗑️ Clear All Messages", variant='secondary')
    def user(user_message, history):
        writehistory(f"USER: {user_message}")
        return "", history + [[user_message, None]]

    def bot(history):
        SYSTEM_PROMPT = """<|im_start|>system
        You are a helpful bot. Your answers are clear and concise.
        <|im_end|>

        """    
        prompt = f"<|im_start|>system<|im_end|><|im_start|>user\n{history[-1][0]}<|im_end|>\n<|im_start|>assistant\n"  
        print(f"history lenght: {len(history)}")
        if len(history) == 1:
            print("this is the first round")
        else:
            print("here we should pass more conversations")
        history[-1][1] = ""
        for character in llm(prompt, 
                 temperature = 0.32,
                 top_p = 0.95, 
                 repetition_penalty = 1.15, 
                 max_new_tokens=1200,
                 stop = ['<|im_end|>'],
                 stream = True):
            history[-1][1] += character
            yield history
        """
        for character in llm(prompt, 
                 temperature = temperature,
                 top_p = top_p, 
                 repetition_penalty = rep_pen, 
                 max_new_tokens=max_length_tokens,
                 stop = ['<|im_end|>'],
                 stream = True):
            history[-1][1] += character
            yield history
        """
        writehistory(f"BOT: {history}")    
    # initially msg.submit()
    submitBtn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch(inbrowser=True)

