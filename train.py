import torch
import os
from huggingface_hub import login
import json
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
import datetime
from trl import SFTTrainer
tqdm.pandas()


def get_config_value(config, key, default_value):
    return config[key] if key in config and config[key] is not None else default_value


with open("config.json", "r") as f:
    config = json.load(f)

# Step 1: Load arguments
print("Step 1: Load arguments")
load_in_4bit = get_config_value(config, "load_in_4bit", True)
load_in_8bit = get_config_value(config, "load_in_8bit", False)
model_name = get_config_value(config, "model_name", "EleutherAI/gpt-neo-125m")
data_dir = get_config_value(config, "data_dir", "./data")
output_dir = get_config_value(config, "output_dir", "./output")
model_max_length = get_config_value(config, "model_max_length", 1024)
batch_size = get_config_value(config, "batch_size", 1)
gradient_accumulation_steps = get_config_value(
    config, "gradient_accumulation_steps", 1)
learning_rate = get_config_value(config, "learning_rate", 0.0001)
logging_steps = get_config_value(config, "logging_steps", 1)
num_train_epochs = get_config_value(config, "num_train_epochs", 3)
max_steps = get_config_value(config, "max_steps", -1)
use_peft = get_config_value(config, "use_peft", False)
peft_lora_r = get_config_value(config, "peft_lora_r", 64)
peft_lora_alpha = get_config_value(config, "peft_lora_alpha", 16)
hf_repo_name = get_config_value(
    config, "hf_repo_name", model_name + "-" + datetime.datetime.now().strftime("%Y_%m_%d"))

# Step 1.1: If hf_token is not None, login to the Hub
hf_token = get_config_value(config, "hf_token", None)
if hf_token:
    print("Step 1.1: Login to the Hub")
    login(token=hf_token)
    print("Logged in to the Hub")

# Step 2: Prepare model config
print("Step 2: Prepare model config")
device_map = None
if load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit)
    torch_dtype = torch.bfloat16
elif load_in_8bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    torch_dtype = torch.bfloat16
else:
    quantization_config = None
    torch_dtype = None

# Step 3: Load the model
print("Step 3: Load the model")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=False,
    torch_dtype=torch_dtype,
)

# Step 4: Load the dataset
print("Step 4: Load the dataset (ignore below tokenizer warning)")
# dataset = load_dataset("text", data_dir=dataset_dir, split="train")
# read dataset_dir
_texts = []
for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), "r") as f:
        _texts.append(f.read())

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Instead of filtering, we'll break texts into chunks
chunks = []
for text in _texts:
    tokenized_text = tokenizer.encode(text)
    for i in range(0, len(tokenized_text), model_max_length):
        chunk = tokenized_text[i:i+model_max_length]
        chunks.append(tokenizer.decode(chunk))

texts = chunks

dataset = Dataset.from_dict({"text": texts})

# Step 5: Define the training arguments
print("Step 5: Define the training arguments")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
)

# Step 6: Define the LoraConfig, if using PEFT
print("Step 6: Define the LoraConfig, if using PEFT")
peft_config = None
if use_peft:
    print("Note: when using PEFT, consider specifying `target_modules` for better performance")
    peft_config = LoraConfig(
        r=peft_lora_r,
        lora_alpha=peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )

# Step 7: Define the Trainer and train the model
print("Step 7: Define the Trainer and train the model")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=model_max_length,
)
trainer.train()

# Step 8: Save the model locally
print("Step 8: Save the model locally")
final_checkpoints_dir = os.path.join(
    output_dir, f"train_checkpoint-on-{datetime.datetime.now().strftime('%Y_%m_%d')}")
trainer.model.save_pretrained(final_checkpoints_dir)

# merge lora weights, if using PEFT
if use_peft:
    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_checkpoints_dir, device_map=device_map, torch_dtype=torch.bfloat16)

    model = model.merge_and_unload()
    output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)

# Step 9: Push the model to the Hub
if hf_token and hf_repo_name:
    print("Step 9: Push the model to the Hub")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.push_to_hub(hf_repo_name)
    model.push_to_hub(hf_repo_name)

print("Done!")

# Load the model locally
# output_dir = "./output"
# final_checkpoints_dir = os.path.join(output_dir, f"train_checkpoint-on-2023_08_10")
# AutoModelForCausalLM.from_pretrained(final_checkpoints_dir, torch_dtype=torch.bfloat16)
