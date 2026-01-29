import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output', help='output dir (in output directory will be created generated/ dir with results)')
    parser.add_argument('-lp', '--lora_path', type=str, default=None, help='path to directory with pre-trained model')

    args = parser.parse_args()

    load_dotenv()

    access_token = os.getenv('HF_TOKEN')
    # Qwen3-4B-Instruct-2507
    model_name = 'Qwen/Qwen3.0-4B'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        token=access_token
    )

    if (args.lora_path is not None):
        model = PeftModel.from_pretrained(model, args.lora_path)

    print(f'Используется {model.device}')

    model.eval()

    topics = ['любовь', 'память', 'природа', 'грусть', 'радость']

    prompts = []

    for topic in topics:
        prompts.append(f'Напиши стихотворение из 2-4 четверостиший на тему: {topic}')

    result = []

    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        result.append({
            'prompt': prompt,
            'generated_text': generated
        })

    out_df = pd.DataFrame(result)
    out_df.to_csv(os.path.join(args.output, 'generated/generation_result.csv'), index=False)

if __name__=='__main__':
    main()
