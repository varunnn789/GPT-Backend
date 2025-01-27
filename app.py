from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Hardcoded model name and token
MODEL_NAME = "singhvarun789/Mental_Health_Fine_Tuned_GPT2"
API_TOKEN = "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"

try:
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=API_TOKEN)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model or tokenizer: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the Mindfulness GPT-2 API!"}


@app.post("/generate")
async def generate_text(prompt: str):
    """
    Generate text based on a given prompt using the fine-tuned GPT-2 model.

    Args:
    - prompt (str): The input prompt for the model.

    Returns:
    - dict: The generated text.
    """
    try:
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate text
        output = model.generate(
            input_ids,
            max_length=200,  # Adjust max length as needed
            temperature=0.7,  # Sampling temperature
            top_k=50,         # Consider top-k tokens
            do_sample=True    # Enable sampling for randomness
        )

        # Decode output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return {"generated_text": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
