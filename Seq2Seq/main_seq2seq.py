from torch.utils.data.dataset import Dataset
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
from util import sampler
from torch.autograd import Variable

import Seq2Seq.dataset_seq2seq as dataset
import Seq2Seq.model_seq2seq as Model
import Seq2Seq.Train_seq2seq as Train
import os
import shutil
import csv

def main():
    #Hyperparameter
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lrlist',default=[0.001,0.0001],help="list for adjust learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=0.0000001,
                        help='weight decay parameter')
    #Data sampling method
    parser.add_argument('--new_version',default=True,type=bool,help="True : longer input length, False : input length + prediction length = 31")
    parser.add_argument('--random-length',default=[False, True], type=bool,help="if you want to use randomized sample length, True.")
    parser.add_argument('--before_lim',default=120,type=int,help='initialize batchs input length')
    parser.add_argument('--after_lim',default=30,type=int,help='initialize batchs prediction length')
    # Mode Configure
    parser.add_argument('--test', default=False, help='use for testing(trained model)')
    parser.add_argument('--save-model', action='store_true', default=True,  # False~>True
                        help='For Saving the current Model')
    parser.add_argument('--name', type=str, default='GRU_S2S_120_pre_x10')
    parser.add_argument('--log-pass',type = int, default=20000)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',  # 10~>50
                        help='how many batches to wait before logging training status')
    parser.add_argument('--isweather',default=False, help='key of using weather information')
    parser.add_argument('--isMSEweighted',default=False,help='key for using weighted MSE')
    #Path
    # parser.add_argument('--resume', default='./model/LSTM_S2S_120/101001checkpoint_lstm_std.pth.tar', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume',default='',type=str)
    parser.add_argument('--pretrain', default=True, type=bool)
    parser.add_argument('--pretrained-model', default='../model/GRU_120_4/14652checkpoint_lstm_std.pth.tar', type=str)
    parser.add_argument("--data-root",default="../data/03_merge/v06_divide_train_test/")
    parser.add_argument("--normalize-factor-path",type=str,default='../data/etc/normalize_factor.csv')
    parser.add_argument("--usable-idx-path",type=str,default="../data/etc/")

    # Cuda Configureration
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Normalize factor load
    fr = open(args.normalize_factor_path, 'r', encoding='cp949', newline='')
    normalize_factor = list(csv.reader(fr))
    if args.test:
        train_loss_dataset = dataset.CustomDataset(args, args.data_root, "v05_trainset.csv", "idx_train_loss.csv",
                                                   for_test=True)
        train_loss_sampler = sampler.BatchSampler(sampler.SequentialSampler(train_loss_dataset),
                                                  batch_size=2000,
                                                  drop_last=False,
                                                  random_length=args.random_length, for_test=True,
                                                  before_lim=args.before_lim,
                                                  after_lim=args.after_lim, new_version_sampler=args.new_version)
        train_loss_loader = torch.utils.data.DataLoader(train_loss_dataset, batch_sampler=train_loss_sampler, **kwargs)

        test_dataset = dataset.CustomDataset(args, args.data_root, "v05_testset.csv", "idx_test.csv", for_test=True)
        test_sampler = sampler.BatchSampler(sampler.SequentialSampler(test_dataset),
                                            batch_size=2000,
                                            drop_last=False,
                                            random_length=args.random_length, for_test=True,
                                            before_lim=args.before_lim,
                                            after_lim=args.after_lim, new_version_sampler=args.new_version)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler, **kwargs)
    else:
        train_dataset = dataset.CustomDataset(args,args.data_root,"v05_trainset.csv","idx_train.csv")
        # BatchSampler Parameter : (sampler, batch_size, drop_last,random_length = False,for_test=False,new_type_sampler=True,before_lim=-1,after_lim=-1)
        train_sampler=sampler.BatchSampler(sampler.RandomSampler(train_dataset),batch_size=args.batch_size,drop_last=False,
                                           random_length=args.random_length,before_lim=args.before_lim,after_lim=args.after_lim,new_version_sampler=args.new_version)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_sampler=train_sampler,**kwargs)

        train_loss_dataset = dataset.CustomDataset(args, args.data_root, "v05_trainset.csv", "idx_train_loss.csv",
                                                   for_test=True)
        train_loss_sampler = sampler.BatchSampler(sampler.RandomSampler(train_loss_dataset),
                                                  batch_size=args.test_batch_size,
                                                  drop_last=False,
                                                  random_length=args.random_length, for_test=True,
                                                  before_lim=args.before_lim,
                                                  after_lim=args.after_lim, new_version_sampler=args.new_version)
        train_loss_loader = torch.utils.data.DataLoader(train_loss_dataset, batch_sampler=train_loss_sampler, **kwargs)

        test_dataset = dataset.CustomDataset(args, args.data_root, "v05_testset.csv", "idx_test.csv", for_test=True)
        test_sampler = sampler.BatchSampler(sampler.RandomSampler(test_dataset),
                                            batch_size=args.test_batch_size,
                                            drop_last=False,
                                            random_length=args.random_length, for_test=True,
                                            before_lim=args.before_lim,
                                            after_lim=args.after_lim, new_version_sampler=args.new_version)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler, **kwargs)

    #Training Configuration
    writer = SummaryWriter('./log/' + args.name + '/')
    print(device)
    encoder = Model.EncoderGRU().to(device)
    decoder = Model.DecoderGRU().to(device)

    if args.isMSEweighted:
        criterion = weighted_mse_loss
    else:
        criterion = nn.MSELoss().to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model= (encoder,decoder)
    optimizer = (encoder_optimizer,decoder_optimizer)
    training = Train.Training(model)
    steps = 0
    best_loss = 1000000
    start_epoch=1

    if args.pretrain:
        if os.path.isfile(args.pretrained_model):
            print("=> loading pre model '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            encoder.load_state_dict(checkpoint['state_dict'])
            encoder_optimizer.load_state_dict(checkpoint["optimizer"])
            print("=> loaded pretrained model '{}' (epoch {})"
                  .format(args.pretrained_model, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained__model))
    #Load Trained Model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            steps= checkpoint['steps']
            best_loss=checkpoint['best_loss']
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if not args.test:
        #학습용
        for param_group in decoder_optimizer.param_groups:
            param_group['lr'] = args.lr
        for epoch in range(start_epoch,args.epochs):
            print(epoch)
            adjust_learning_rate(decoder_optimizer,steps,args)
            adjust_learning_rate(encoder_optimizer, steps, args)
            steps,best_loss = training.train(args,encoder,decoder,criterion,device,train_loader,test_loader,train_loss_loader,optimizer,epoch,writer,normalize_factor,steps,best_loss)
    else:
        #검증용
        print("activate test code!!!!")
        args.random_length[0] = False
        test_mse, test_mae, test_won_sum_mae, test_won_mae,test_mae_list,test_var= training.eval(args,test_loader,encoder,decoder,criterion,device,normalize_factor,teacher_forcing=False)
        writer.add_scalars('Loss_test', { 'test loss': test_mae}, 1)
        writer.add_scalars('Won mae_test',{'test_won_mae':test_won_sum_mae},1)
        writer.add_scalars('energy_prediction_mae_test', {'elec': test_won_mae[0], 'water': test_won_mae[1], 'gas': test_won_mae[2]},1)
        print("mae_won={}".format(test_won_sum_mae))

        for i in range(30):

            writer.add_scalars('예측 길이에 따른 오차_test', {'test won': test_mae_list[i]}, i+1)

def adjust_learning_rate(optimizer, steps,args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    i = steps // 65000
    try:
        lr=args.lrlist[i]
    except:
        lr=args.lrlist[-1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(lr)

def weighted_mse_loss(input,target):
    weights = Variable(torch.Tensor([2.14,0.66,0.20]).cuda())#전기 평균요금 : 49179, 수도 : 15123, 가스 : 4557 -> 2.14,0.66,0.2
    pct_var = (input-target)**2
    out = pct_var*weights.expand_as(target)
    loss = out.mean()
    return loss

if __name__ == '__main__':
    main()