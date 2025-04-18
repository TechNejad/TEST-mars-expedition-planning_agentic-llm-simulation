# LLM Models I'm Using for My Mars Expedition Simulation Project

After spending way too much time searching through Hugging Face (and a bunch of Reddit threads), I think I've found some decent open-source models that should work on my free Colab account:

## Main Choice: Llama 3 8B

- **Model**: [unsloth/llama-3-8b](https://huggingface.co/unsloth/llama-3-8b)
- **Size**: Around 8B parameters (pretty big but still runs on Colab)
- **Why I chose it**: 
  - It's the newest Meta model I could find
  - Works okay for dialogue stuff
  - Most importantly - it actually runs on Colab's free tier with the T4 GPU!
  - I found this helpful Colab notebook that shows how to use it: [https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp)

## Backup Plan: Llama 2 7B Chat

If Llama 3 doesn't work out (or if I run into GPU memory issues), I might try:
- **Model**: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- **Size**: 7B parameters
- **Why it might work**:
  - It's a bit smaller than Llama 3
  - Lots of people have used it for chat stuff
  - Tons of examples online for how to use it

## How I'm Planning to Implement This

My current plan (might change as I figure things out):

1. Try the Llama 3 8B model first
2. Use the Transformers library (seems easiest)
3. Figure out how to connect it with the Inspect framework
4. Set up the two agent roles from my project spec
5. Try to add that "internal reasoning" thing my professor mentioned

Here's some basic code I'm starting with (copied from a tutorial but I'm still figuring out how it all works):

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "unsloth/llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # this helps with memory issues
    device_map="auto"
)

# Basic function to generate text
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9  # still not sure what these parameters really do
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

I'm not 100% sure this will work for my project, but it's a starting point. I'll probably need to modify it a lot as I go.
