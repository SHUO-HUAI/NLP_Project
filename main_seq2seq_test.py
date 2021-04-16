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
from classes import Dictionary, CountDictionary
# import model
from functions import to_cuda
from preprocessing import read_files, prepare_data, prepare_summary, zero_pad, remove_pad, prepare_dictionary, \
    prepare_art_sum, test_dic, test_train
from model import Model
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
parser.add_argument('--lr', '--learning-rate', default=config.lr, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

best_acc1 = 0.0


#
# dic_tmp = test_dic()
# # prepare_art_sum
# test_train(dic_tmp)
# exit()

def data_process(args):
    os.mkdir(args.load_data)
    split_file = args.split_file

    test_path = split_file['test']
    train_path = split_file['train']
    valid_path = split_file['val']

    dic_path = os.path.join(args.load_data, 'dictionary')
    out_path = args.load_data

    # dic = prepare_train_art_sum(train_path, dic_path, out_path)
    dic = prepare_dictionary(train_path, dic_path)
    prepare_art_sum(train_path, os.path.join(out_path, 'train_set'), dic)
    prepare_art_sum(valid_path, os.path.join(out_path, 'valid_set'), dic)
    prepare_art_sum(test_path, os.path.join(out_path, 'test_set'), dic)


def data_loader(args):
    args.batch_size = 1
    dic_path = os.path.join(args.load_data, 'dictionary')
    args.load_data = './'
    out_path = args.load_data

    with open(dic_path, 'rb') as f:
        dic = pickle.load(f)

    with open(os.path.join(out_path, 'train'), 'rb') as f:
        padded_train = pickle.load(f)

    with open(os.path.join(out_path, 'train'), 'rb') as f:
        padded_valid = pickle.load(f)

    with open(os.path.join(out_path, 'train'), 'rb') as f:
        padded_test = pickle.load(f)

    print('Train size:', len(padded_train))
    print('Valid size:', len(padded_valid))
    print('Test size:', len(padded_test))
    print('Directory size', len((dic.word2idx.keys())))

    # for key in dic.word2idx.keys():
    #     print(key,dic.word2idx[key])

    return padded_train, padded_valid, padded_test, dic


def validate(val_set, model, args):
    model.eval()
    print_count = 0

    data_size = len(val_set)
    start_tmp = 0
    args.batch_size = 5
    batch_num = math.ceil(data_size / args.batch_size)
    acc_all = 0.0

    with torch.no_grad():
        while start_tmp < data_size:

            print_count += 1
            if start_tmp + args.batch_size < data_size:
                cur_batch = val_set[start_tmp: start_tmp + args.batch_size]
                start_tmp = start_tmp + args.batch_size
            else:
                cur_batch = val_set[start_tmp:]
                start_tmp = start_tmp + args.batch_size

            padded_articles = cur_batch[:, 0]
            padded_summaries = cur_batch[:, 1]

            padded_articles = np.array([np.array(tmp) for tmp in padded_articles])
            padded_summaries = np.array([np.array(tmp) for tmp in padded_summaries])

            tensor_art = torch.LongTensor(padded_articles.astype(np.float32))
            tensor_sum = torch.LongTensor(padded_summaries.astype(np.float32))

            tensor_art = to_cuda(tensor_art)
            tensor_sum = to_cuda(tensor_sum)

            out_list, _ = model(tensor_art, None, False)

            output_list = []
            target_list = []

            for j in range(out_list.shape[0]):
                k = remove_pad(tensor_sum[j, 1:])
                # print(out_list[j, :k].shape)
                out_tmp = ' '.join(map(str, torch.argmax(out_list[j, :k - 1], 1).cpu().numpy()))
                tar_tmp = ' '.join(map(str, (tensor_sum[j, 1:k]).cpu().numpy()))

                output_list.append(out_tmp)
                target_list.append(tar_tmp)

                out_tmp = torch.argmax(out_list[j, :k], 1).cpu().numpy()
                tar_tmp = (tensor_sum[j, :k]).cpu().numpy()
                #
                # output_list.append(out_tmp)
                # target_list.append(tar_tmp)
                # print(out_tmp)
                # print(tar_tmp)
                print('*************************************************************************')

                out_string = []
                for word in out_tmp:
                    out_string.append(model.dictionary.idx2word[word])
                print(out_string)

                out_string = []
                for word in tar_tmp:
                    out_string.append(model.dictionary.idx2word[word])
                print(out_string)
                print('*************************************************************************')

            acc = accuracy(output_list, target_list)
            print('Test [' + str(print_count) + '/' + str(batch_num) + ']', 'Acc ', acc)

            acc_all = acc_all + acc * len(cur_batch)
        acc_all = acc_all / data_size
        print(' *Accuracy all:', acc_all)
    args.batch_size = 1
    return acc_all


def test(val_set, model):
    model.eval()
    batch_size = 1
    data_size = len(val_set)
    # batch_num = math.ceil(data_size / batch_size)
    start_tmp = 0
    with torch.no_grad():
        while start_tmp < data_size:

            if start_tmp + batch_size < data_size:
                cur_batch = val_set[start_tmp: start_tmp + batch_size]
                start_tmp = start_tmp + batch_size
            else:
                cur_batch = val_set[start_tmp:]
                start_tmp = start_tmp + batch_size

            padded_articles = cur_batch[:, 0]
            padded_summaries = cur_batch[:, 1]

            padded_articles = np.array([np.array(tmp) for tmp in padded_articles])
            padded_summaries = np.array([np.array(tmp) for tmp in padded_summaries])

            tensor_art = torch.LongTensor(padded_articles.astype(np.float32))
            tensor_sum = torch.LongTensor(padded_summaries.astype(np.float32))

            tensor_art = to_cuda(tensor_art)
            tensor_sum = to_cuda(tensor_sum)

            out_list, _ = model(tensor_art, None, False)

            output_list = []
            target_list = []

            for j in range(out_list.shape[0]):
                k = remove_pad(tensor_sum[j, :])
                # print(out_list[j, :k].shape)
                out_tmp = torch.argmax(out_list[j, :k], 1).cpu().numpy()
                tar_tmp = (tensor_sum[j, :k]).cpu().numpy()

                output_list.append(out_tmp)
                target_list.append(tar_tmp)
                print(out_tmp)
                print(tar_tmp)
                input()
        #
        #     acc = accuracy(output_list, target_list)
        #     print('Test [' + str(print_count) + '/' + str(batch_num) + ']', 'Acc ', acc)
        #
        #     acc_all = acc_all + acc * len(cur_batch)
        # acc_all = acc_all / data_size
        # print(' *Accuracy all:', acc_all)
    # return acc_all


def train(train_set, model, criterion, optimizer, epoch, args):
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
            start_tmp = start_tmp + args.batch_size

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
            # k = remove_pad(tensor_sum[j, :])
            # k = min(k, len(out_list[j]), len(tensor_sum[j]))

            # k = remove_pad(tensor_sum[j, :])
            # print(out_list[j, :k].shape)

            k = remove_pad(tensor_sum[j, 1:])
            k = min(k, len(out_list[j]), len(tensor_sum[j]))
            # k = min(k, len(out_list[j]), len(tensor_sum[j, 1:]))

            out_tmp = torch.argmax(out_list[j, :k - 1], 1).cpu().numpy()
            tar_tmp = (tensor_sum[j, 1:k]).cpu().numpy()
            #
            # output_list.append(out_tmp)
            # target_list.append(tar_tmp)
            # print(out_tmp)
            # print(tar_tmp)

            out_string = []
            for word in out_tmp:
                out_string.append(model.dictionary.idx2word[word])
            print(out_string)

            out_string = []
            for word in tar_tmp:
                out_string.append(model.dictionary.idx2word[word])
            print(out_string)

            # input()

            # loss += criterion(torch.log(out_list[j, :k]), tensor_sum[j, :k])
            loss += criterion(torch.log(out_list[j, :k - 1]), tensor_sum[j, 1:k])

        loss += cov_loss

        # if print_count % 100 == 0:
        print('Epoch: [' + str(epoch) + '] [' + str(print_count) + '/' + str(batch_num) + ']', 'Loss ',
              loss.cpu().detach().numpy(), 'cov Loss', cov_loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()
    print('End of Epoch', epoch, 'time cost', time.time() - end)


def accuracy(output, target):
    print(output)
    print(target)
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


def adjust_learning_rate(optimizer, epoch, args):
    # ad = int(args.epochs / 3)
    if epoch == 0:
        lr = args.lr
    else:
        lr = max(args.lr / (epoch * 2), args.lr / 10.0)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    global best_acc1
    args = parser.parse_args()
    load_data_floder = args.load_data

    if_load = os.path.exists(load_data_floder)
    if not if_load:
        data_process(args)

    assert os.path.exists(load_data_floder)

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # train_set = [[train_art1, train_sum1],[train_art2, train_sum2], ...] and val_set, test_set
    train_set, val_set, test_set, dic = data_loader(args)

    # articles_len = len(train_set[0][0])

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

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
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
        test(test_set, model)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate ?????

        adjust_learning_rate(optimizer, epoch, args)

        train(train_set, model, criterion, optimizer, epoch, args)
        acc1 = validate(val_set, model, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # is_best = True
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)


if __name__ == '__main__':
    main()
