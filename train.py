import torch
import os
from huggingface_hub import login
import json
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer, AutoPeftModelForCausalLM
import datetime
from trl import SFTTrainer
tqdm.pandas()


def get_config_value(config, key, default_value):
    return config[key] if key in config and config[key] is not None else default_value


with open("config.json", "r") as f:
    config = json.load(f)

# Step 1: Load arguments
print("Step 1: Load arguments")
load_in_8bit = get_config_value(config, "load_in_8bit", True)
load_in_4bit = get_config_value(config, "load_in_4bit", False)
model_name = get_config_value(config, "model_name", "gpt2")
dataset_dir = get_config_value(config, "dataset_dir", "./data")
output_dir = get_config_value(config, "output_dir", "./output")
batch_size = get_config_value(config, "batch_size", 64)
gradient_accumulation_steps = get_config_value(
    config, "gradient_accumulation_steps", 16)
learning_rate = get_config_value(config, "learning_rate", 1.41e-5)
logging_steps = get_config_value(config, "logging_steps", 1)
num_train_epochs = get_config_value(config, "num_train_epochs", 3)
max_steps = get_config_value(config, "max_steps", -1)
use_peft = get_config_value(config, "use_peft", False)
peft_lora_r = get_config_value(config, "peft_lora_r", 64)
peft_lora_alpha = get_config_value(config, "peft_lora_alpha", 16)
hf_token = get_config_value(config, "hf_token", None)
hf_repo_name = get_config_value(
    config, "hf_repo_name", model_name + "-" + datetime.datetime.now().strftime("%Y_%m_%d"))


# Step 2: Prepare model config
print("Step 2: Prepare model config")
device_map = "auto"
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
print("Step 4: Load the dataset")
dataset = load_dataset("text", data_dir=dataset_dir)

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
)
trainer.train()

# Step 8: Save the model locally
print("Step 8: Save the model locally")
final_checkpoints_dir = os.path.join(output_dir, "train_checkpoints")
trainer.model.save_pretrained(final_checkpoints_dir)

# Free memory for merging weights
del model
torch.cuda.empty_cache()
model = AutoPeftModelForCausalLM.from_pretrained(
    final_checkpoints_dir, device_map=device_map, torch_dtype=torch.bfloat16)

# merge lora weights, if using PEFT
if use_peft:
    model = model.merge_and_unload()
output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

# Step 9: Push the model to the Hub
if hf_token:
    print("Step 9: Push the model to the Hub")
    login(token="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")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.push_to_hub(hf_repo_name)
model.push_to_hub(hf_repo_name)

print("Done!")
