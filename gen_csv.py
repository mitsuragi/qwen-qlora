import csv
import argparse
import os
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='input file')
    parser.add_argument('--output', type=str, default='dataset', help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--train_ratio', type=float, default=0.9)

    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    poems = []
    with open(args.input, 'r', encoding='utf-8') as f:
        buffer = []

        for line in f:
            line = line.rstrip('\n')

            if line == '<EOP>':
                if buffer:
                    title = buffer[0]

                    poem = '\n'.join(buffer[1:]).strip()
                    poems.append({
                        'title': title,
                        'poem': poem
                    })

                    buffer = []
            else:
                buffer.append(line)

    random.shuffle(poems)

    split_idx = int(len(poems) * args.train_ratio)
    train_data = poems[:split_idx]
    val_data = poems[split_idx:]

    def save_csv(path, data):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'poem'])
            writer.writeheader()
            writer.writerows(data)

    save_csv(os.path.join(args.output, 'train.csv'), train_data)
    save_csv(os.path.join(args.output, 'val.csv'), val_data)
    print(len(poems))
    print(len(train_data))
    print(len(val_data))

if __name__ == '__main__':
    main()
