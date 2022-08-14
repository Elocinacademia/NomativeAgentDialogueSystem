import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from operator import itemgetter

import dataloader
from model import MLPmodel


def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

if __name__ == "__main__":
    arglist = []
    parser = argparse.ArgumentParser(description='PyTorch Rule Mining')
    parser.add_argument('--data', type=str, default='./data/',
                        help='location of the data corpus')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to embeddings (0 = no dropout)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--seed', type=int, default=999,
                        help='random seed')
    parser.add_argument('--output', type=str, default='./exp/',
                        help='location of the experiment')
    parser.add_argument('--logfile', type=str, default='./exp/log.txt',
                        help='location of the log file')
    args = parser.parse_args()

    ### Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    ### Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Load data
    train_loader, val_loader, test_loader, datatype_dict, recipient_dict, condition_dict = dataloader.create(
        args.data, batchSize=args.batch_size, workers=0)

    ### Model
    model = MLPmodel(args, len(datatype_dict.idx2obj), len(recipient_dict.idx2obj), len(condition_dict.idx2obj))
    model.to(device)

    ### Optimiser
    lr = args.lr

    ### Criterion
    criterion = nn.CrossEntropyLoss()

    ### NEWBOB
    best_val_acc = 0

    for n in range(args.epochs):
        # Start training loop
        model.train()
        total_loss = 0
        total_sample = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wdecay)
        for i, minibatch in enumerate(train_loader):
            feature, label = minibatch
            feature = feature.to(device)
            label = label.to(device)
            output = model(feature)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * label.size(0)
            total_sample += label.size(0)
            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / total_sample
                logging("Epoch: {} | Step: {} out of {} | Loss: {:.5f}".format(n, i, len(train_loader), cur_loss))

        total_val_loss = 0
        total_val_sample = 0
        total_val_corr = 0
        model.eval()
        with torch.no_grad():
            for i, minibatch in enumerate(val_loader):
                feature, label = minibatch
                feature = feature.to(device)
                label = label.to(device)
                output = model(feature)
                loss = criterion(output, label)
                selected = torch.max(output, dim=-1)[1]
                corr = (selected == label).sum()
                total_val_corr += corr
                total_val_loss += loss * label.size(0)
                total_val_sample += label.size(0)
        logging('='*89)
        logging("Epoch: {} | Validation Loss: {:.5f} | Validation Accuracy: {:.2f}".format(n, total_val_loss/total_val_sample, total_val_corr/total_val_sample))
        logging('='*89)
        if total_val_corr/total_val_sample > best_val_acc:
            logging("saving best model...")
            torch.save(model.state_dict(), os.path.join(args.output, 'model.state_dict'))
            best_val_acc = total_val_corr / total_val_sample
