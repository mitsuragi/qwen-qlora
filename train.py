from dataset import get_dataloaders
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling, training_args
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model 
import os

def print_trainable_parameters(model):
    trainable_params = 0 
    all_param = 0 
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f'trainable params: {trainable_params} | all params: {all_param} | trainable% {100 * trainable_params / all_param}')

def main():
    load_dotenv()

    access_token = os.getenv('HF_TOKEN')
    model_name = 'Qwen/Qwen3-0.6B'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(   
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        token=access_token,
    )
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    train_dl, val_dl = get_dataloaders('dataset', tokenizer)

    training_args = TrainingArguments(
        output_dir='output',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=100,
        eval_steps=50,
        save_total_limit=2,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dl.dataset,
        eval_dataset=val_dl.dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    model.save_pretrained('output')

if __name__ == '__main__':
    main()
