from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os
import torch
from datasets import load_dataset
from datasets import DatasetDict, Dataset
from trl import SFTTrainer
import pandas as pd

#### Example of script that we used for finetuning our models. This in particualr is used for finetuning Mistral on Prosocial-Dialog Dataset.

base_model = 'filipealmeida/Mistral-7B-Instruct-v0.1-sharded'
dataset_name = "allenai/prosocial-dialog"

bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)

model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True


dataset = load_dataset(dataset_name, split="train")

texts = [f"<s>[INST] {dataset['context'][i]} [/INST] {dataset['response'][i]} </s>"
        for i in range(2500)]

texts = dataset['text'].to_list()
data = [{'text' : text} for text in texts[:2000]]
preprocessed_dataset = Dataset.from_list(data)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Define the LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# Apply the PEFT model
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./results_moralstories",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=700,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=preprocessed_dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()
trainer.save_model("finetuned_weights/fine-tuned-model")
tokenizer.save_pretrained("finetuned_weights/fine-tuned-model")


