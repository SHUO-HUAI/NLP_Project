import argparse
import math
import os
import pickle
from os import listdir
from os.path import isfile, join
from io import open
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
import sys
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time
from classes import Dictionary
import model
from functions import to_cuda
from preprocessing import read_files, prepare_data, prepare_summary, zero_pad, remove_pad, prepare_train_art_sum, \
    prepare_valid_art_sum
from model_all import Model
import config
import shutil
from rouge import Rouge

parser = argparse.ArgumentParser(description='PyTorch Get To The Point Training')
# parser.add_argument('--path', type=list, default=None,
#                     help='path to the training data')
parser.add_argument('--split_file', type=dict, default=config.split_file, help='path to the training data')
parser.add_argument('--load_data', type=str, default=config.load_data, help='path to the data after preprocessing')
parser.add_argument('--resume', type=str, default=config.resume, help='path to the resume checkpoint')
parser.add_argument('--epochs', type=int, default=config.epoch, help='path to the resume checkpoint')
parser.add_argument('--save', type=str, default=config.save, help='path to the resume checkpoint')
parser.add_argument('--batch_size', type=int, default=config.batch_size, help='path to the resume checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on validation set')

best_acc1 = 0.0


def data_process(args):
    os.mkdir(args.load_data)
    split_file = args.split_file

    test_path = split_file['test']
    train_path = split_file['train']
    valid_path = split_file['val']

    dic_path = os.path.join(args.load_data, 'dictionary')
    out_path = args.load_data

    dic = prepare_train_art_sum(train_path, dic_path, out_path)
    prepare_valid_art_sum(valid_path, os.path.join(out_path, 'valid_set'), dic)
    prepare_valid_art_sum(test_path, os.path.join(out_path, 'test_set'), dic)


def data_loader(args):

    dic_path = os.path.join(args.load_data, 'dictionary')
    out_path = args.load_data

    with open(dic_path, 'rb') as f:
        dic = pickle.load(f)

    with open(os.path.join(out_path, 'train_set'), 'rb') as f:
        padded_train = pickle.load(f)

    with open(os.path.join(out_path, 'valid_set'), 'rb') as f:
        padded_valid = pickle.load(f)

    with open(os.path.join(out_path, 'test_set'), 'rb') as f:
        padded_test = pickle.load(f)

    print('Train size:', len(padded_train))
    print('Valid size:', len(padded_valid))
    print('Test size:', len(padded_test))

    return padded_train, padded_valid, padded_test, dic


def validate(val_set, model, args):
    model.eval()
    print_count = 0

    data_size = len(val_set)
    start_tmp = 0
    batch_num = math.ceil(data_size / args.batch_size)

    with torch.no_grad():
        while start_tmp < data_size:

            print_count += 1
            if start_tmp + args.batch_size < data_size:
                cur_batch = val_set[start_tmp: start_tmp + args.batch_size]
                start_tmp = start_tmp + args.batch_size
            else:
                cur_batch = val_set[start_tmp:]

            padded_articles = cur_batch[:, 0]
            padded_summaries = cur_batch[:, 1]

            padded_articles = np.array([np.array(tmp) for tmp in padded_articles])
            padded_summaries = np.array([np.array(tmp) for tmp in padded_summaries])

            tensor_art = torch.LongTensor(padded_articles.astype(np.float32))
            tensor_sum = torch.LongTensor(padded_summaries.astype(np.float32))

            tensor_art = to_cuda(tensor_art)
            tensor_sum = to_cuda(tensor_sum)

            out_list, _ = model(tensor_art, tensor_sum)

            output_list = []
            target_list = []

            for j in range(out_list.shape[0]):
                k = remove_pad(tensor_sum[j, 1:])
                out_tmp = ' '.join(map(str, torch.argmax(out_list[j, :k - 1], 1).cpu().numpy()))
                tar_tmp = ' '.join(map(str, torch.argmax(tensor_sum[j, 1:k], 1).cpu().numpy()))

                output_list.append(out_tmp)
                target_list.append(tar_tmp)

            if print_count % 100 == 0:
                print('Test [' + str(print_count) + '/' + str(batch_num) + ']')

        acc = accuracy(output_list, target_list)
    return acc


def train(train_set, model, criterion, optimizer, epoch, dic, args):
    model.train()
    print('Start of Epoch: ', epoch)
    print_count = 0
    end = time.time()

    np.random.shuffle(train_set)

    data_size = len(train_set)
    start_tmp = 0
    batch_num = math.ceil(data_size / args.batch_size)

    while start_tmp < data_size:

        print_count += 1
        if start_tmp + args.batch_size < data_size:
            cur_batch = train_set[start_tmp: start_tmp + args.batch_size]
            start_tmp = start_tmp + args.batch_size
        else:
            cur_batch = train_set[start_tmp:]

        padded_articles = cur_batch[:, 0]
        padded_summaries = cur_batch[:, 1]

        padded_articles = np.array([np.array(tmp) for tmp in padded_articles])
        padded_summaries = np.array([np.array(tmp) for tmp in padded_summaries])

        # print(padded_articles)
        # for i in padded_articles:
        #     print(len(i))

        tensor_art = torch.LongTensor(padded_articles.astype(np.float32))
        tensor_sum = torch.LongTensor(padded_summaries.astype(np.float32))

        tensor_art = to_cuda(tensor_art)
        tensor_sum = to_cuda(tensor_sum)

        optimizer.zero_grad()

        out_list, cov_loss = model(tensor_art, tensor_sum)

        loss = torch.tensor(0.)
        loss = to_cuda(loss)

        for j in range(out_list.shape[0]):
            k = remove_pad(tensor_sum[j, 1:])

            loss += criterion(torch.log(out_list[j, :k - 1]), tensor_sum[j, 1:k])

        loss += cov_loss

        if print_count % 100 == 0:
            print('Epoch: [' + str(epoch) + '] [' + str(print_count) + '/' + str(batch_num) + ']', 'Loss ', loss)

        loss.backward()
        optimizer.step()
    print('End of Epoch', epoch, 'time cost', time.time() - end)


def accuracy(output, target):
    rouge = Rouge()
    rouge_score = rouge.get_scores(output, target, avg=True)
    r1 = rouge_score["rouge-1"]['r']
    r2 = rouge_score["rouge-2"]['r']
    rl = rouge_score["rouge-l"]['r']
    return (r1 + r2 + rl) / 3


def save_checkpoint(state, is_best, folder):
    filename = os.path.join(folder, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))


def main():
    global best_acc1
    args = parser.parse_args()
    load_data_floder = args.load_data

    if_load = os.path.exists(load_data_floder)
    if not if_load:
        data_process(args)

    assert os.path.exists(load_data_floder)

    # train_set = [[train_art1, train_sum1],[train_art2, train_sum2], ...] and val_set, test_set
    train_set, val_set, test_set, dic = data_loader(args)

    articles_len = len(train_set[0][0])

    # print(train_set.shape)
    # print(val_set.shape)
    # print(test_set.shape)
    #
    # print(len(train_set[0][0]),len(val_set[0][0]),len(test_set[0][0]))
    #
    # print(train_set[0])
    #
    # print(articles_len)
    # exit()

    # Model
    model = Model(dic)
    model = to_cuda(model)

    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    args.start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_set, model, args)
        return
    if args.test:
        validate(test_set, model, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate ?????
        train(train_set, model, criterion, optimizer, epoch, dic, args)
        acc1 = validate(val_set, model, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)


if __name__ == '__main__':
    main()
