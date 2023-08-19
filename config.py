# Folders of the data to train on and to save the model to
data_dir = "../data"
output_dir = "../output"

# HuggingFace credentials
hf_token = None
hf_repo_name = None # to upload to the Hub after training

# Model name and parameters
model_name = "EleutherAI/gpt-neo-125m"
load_in_4bit = False # load in 4-bit precision, recommended to reduce required GPU memory
load_in_8bit = False 

# Parameter efficient fine tuning (PEFT) arguments
use_peft = True 
peft_lora_r = 16
peft_lora_alpha = 16
target_modules = None

# Training arguments
model_max_length = 1024
learning_rate = 0.0001
num_train_epochs = 3
logging_steps = 1
batch_size = 1
gradient_accumulation_steps = 1