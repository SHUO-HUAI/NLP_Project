import argparse
import numpy as np
import config
from preprocessing import tokenize_stories
import os

parser = argparse.ArgumentParser(description='PyTorch Get To The Point Training')
parser.add_argument('--path', type=str, default=None,
                    help='path to the training data')
parser.add_argument('--token_path', type=str, default=config.all_tokenized_path,
                    help='path to the training data')

args = parser.parse_args()

need_tokenize = not os.path.exists(args.token_path)
if need_tokenize:
    os.mkdir(args.token_path)
    for folder in os.listdir(args.path):
        os.mkdir(os.path.join(args.token_path, folder))
        tokenize_stories(os.path.join(args.path, folder), os.path.join(args.token_path, folder))

all_file_names = []
for folder in os.listdir(args.token_path):
    tmp = os.path.join(args.token_path, folder)
    if os.path.isdir(tmp):
        for file in os.listdir(tmp):
            all_file_names.append(os.path.join(tmp, file))

all_file_names = np.array(all_file_names)
perm = np.random.permutation(all_file_names.shape[0])
# print(all_file_names)
# print(len(all_file_names))

train_file_names = all_file_names[perm[:287227]]
valid_file_names = all_file_names[perm[287227:287227 + 13368]]
test_file_names = all_file_names[perm[287227 + 13368:]]

with open('train_all.txt', 'w+') as f:
    for line in train_file_names:
        f.write(str(line) + '\n')

with open('test_all.txt', 'w+') as f:
    for line in test_file_names:
        f.write(str(line) + '\n')

with open('val_all.txt', 'w+') as f:
    for line in valid_file_names:
        f.write(str(line) + '\n')
