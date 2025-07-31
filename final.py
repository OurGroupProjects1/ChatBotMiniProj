import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import flask
from flask import Flask,request,render_template
import webbrowser

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CodeRunner

print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))

model_path = "../DeepSeek-Coder"
model =  AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    offload_folder="offload")
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
print("model loaded")
target_file = "CodeRunner.py"

def GenerateOutput(message):
    prompt = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
        )
    attention_mask = (prompt != tokenizer.pad_token_id).long()
    
    prompt = prompt.to(model.device)
    attention_mask = attention_mask.to(model.device)    
    outputs = model.generate(
            input_ids=prompt,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            #top_k=50,
            #repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][prompt.shape[1]:], skip_special_tokens=True)
    return response

def CheckForKeyWords(user_input):
    user_input_lower = user_input.lower()
    keywords = []

    if "table" in user_input_lower:
        keywords.append("table")
    if "flowchart" in user_input_lower or "flow chart" in user_input_lower:
        keywords.append("flowchart")
    if "code" in user_input_lower:
        keywords.append("code")
    if any(word in user_input_lower for word in ["explain", "explanation", "in detail"]):
        keywords.append("explain")

    return keywords


def extract_code_block(text):
    if os.path.exists(target_file):
        os.remove(target_file)
    open(target_file, "w").close()

    try:
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if "```" in line or "python" in line)
        end = next(i for i, line in enumerate(lines[start + 1:], start + 1) if "```" in line)
        extracted = "\n".join(lines[start + 1:end])
    except StopIteration:
        extracted = ""

    with open(target_file, "w") as tgt:
        tgt.write(extracted)
    
    return extracted.strip()


def UserInputs(messages):
    user_input = messages[0]["content"]
    keywords = CheckForKeyWords(user_input)

    if not keywords:
        return "it's out of my scope currently"

    if "code" in keywords:
        prompt = f"Generate the code for {user_input}"
    elif "flowchart" in keywords:
        prompt = f"Generate a Graphviz flowchart in Python using the 'graphviz' library for: {user_input}"
    elif "explain" in keywords:
        prompt = f"Explain in detail: {user_input}"
    elif "table" in keywords:
        prompt = f"Generate an HTML table for: {user_input}"
    
    response = GenerateOutput([{"role": "user", "content": prompt}])
    return extract_code_block(response)

app = Flask(__name__)


@app.route("/",methods = ['GET','POST'])
def Input():
    
    prompt_in = request.form.get('result')
    if prompt_in:
        message = [{"role": "user", "content": prompt_in}]
        response1 = UserInputs(message)
        return render_template('frontend.html', prompt=prompt_in, response=response1)
    else:
        return render_template('frontend.html', prompt="", response="")

    
if __name__ == "__main__":
    """
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception as e:
        print("Browser error:", e)"""
    app.run(port=5000)


