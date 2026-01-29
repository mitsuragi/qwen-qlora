import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
import os

class PoetryDataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        tokenizer,
        max_length = 512,
        use_title = False
    ):
        self.data = pd.read_csv(path_to_csv)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_title = use_title

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        poem = row['poem'].strip()

        # if self.use_title and 'title' in row:
        #     text = f'{row['title']}\n\n{poem}'
        # else:
        #     text = poem

        text_instruction = (
            f'Instruction:\nНапиши стихотворение в стиле Анны Ахматовой.\n\n'
            f'Response:\n'
        )

        text = text_instruction + poem + tokenizer.eos_token

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt',
            add_special_tokens=True
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()

        instruction_len = len(
            self.tokenizer(text_instruction, add_special_tokens=False)['input_ids']
        )

        labels[:instruction_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def get_datasets(
    input_dir,
    tokenizer,
    # batch_size=1,
):
    train_dataset = PoetryDataset(
        os.path.join(input_dir, 'train.csv'),
        tokenizer=tokenizer
    )
    val_dataset = PoetryDataset(
        os.path.join(input_dir, 'val.csv'),
        tokenizer=tokenizer
    )

    return train_dataset, val_dataset

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False
    # )
    #
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=data_collator
    # )
    #
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=data_collator
    # )
    #
    # return train_dataloader, val_dataloader


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from dotenv import load_dotenv 
    import os

    load_dotenv()

    access_token = os.getenv('HF_TOKEN')
    model_name = 'Qwen/Qwen3-0.6B'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    dataset = PoetryDataset(
        pd.read_csv('dataset/raw_data.csv'),
        tokenizer=tokenizer
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator
    )

    batch = next(iter(dataloader))

    # print(batch.keys())
    ids = batch["input_ids"][0]

    print("BOS token id:", tokenizer.bos_token_id)
    print("EOS token id:", tokenizer.eos_token_id)

    print("First token:", ids[0].item())
    print("Last token:", ids[-1].item())

    # print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
