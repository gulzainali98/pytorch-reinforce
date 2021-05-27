from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate
from transformer_discriminator import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from transformer_discriminator import *

from rewards import compute_reward
from rewards import *
import vsum_tools
from adversary import *
from distribution_learner import *


ntokens = 64 # the size of vocabulary
emsize = 1024 # embedding dimension
nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
device=torch.device("cuda")

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']")

# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
parser.add_argument('--num_networks', type=int, default='3', help="Number of networks in Ensemble")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=60, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

S_D=TransformerModel(ntokens, emsize, nhead, nhid, nlayers, device,dropout=dropout).to(device)
epoch=140
PATH_D="./rl_models/sd_epoch_"+str(epoch)+".pth"
# PATH= "./saved_models/test_model.pth"
S_D.load_state_dict(torch.load(PATH_D))
S_D.eval()


torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False




ntokens = 64 # the size of vocabulary
emsize = 1024 # embedding dimension
nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

def complete_video(sd, ntokens=1518, device=None, save_seq= None):
    if sd.shape[1]==ntokens:
        return sd
    if sd.shape[1]<ntokens:
        # add more vectors
        fea= np.load("./blank_features.npy",allow_pickle=True)
        # new_tensor= torch.zeros((sd.shape[0],sd.shape[1],50))
        # new_tensor= torch.zeros(sd.shape)
        # diff= sd.shape[0]
        fea= np.expand_dims(fea,axis=0)
        fea= np.expand_dims(fea,axis=0)
        fea= torch.tensor(fea).to(device)
        # print(fea.shape)

        # print(fea.shape)
        while sd.shape[1]!=ntokens:
        # sd= torch.cat([sd,fea],dim=0)
            sd= torch.cat((sd,fea),dim=1)
        # print(sd.shape)
        return sd

    if sd.shape[1]>ntokens:
        try:
            return sd[:,0:ntokens,:]
        except Exception as e:
            print("return saved sequence")
            print(save_seq.shape)
            print(sd.shape)
            if save_seq.shape[1]==ntokens:
                return save_seq
            elif save_seq.shape[1]<ntokens:
                fea= np.load("./blank_features.npy",allow_pickle=True)
                # new_tensor= torch.zeros((sd.shape[0],sd.shape[1],50))
                # new_tensor= torch.zeros(sd.shape)
                # diff= sd.shape[0]
                fea= np.expand_dims(fea,axis=0)
                fea= np.expand_dims(fea,axis=0)
                fea= torch.tensor(fea).to(device)

                while save_seq.shape[1]!=ntokens:
                # sd= torch.cat([sd,fea],dim=0)
                    save_seq= torch.cat((save_seq,fea),dim=1)
                # print(sd.shape)
                return save_seq
            else:
                return save_seq[:,0:ntokens,:]


def cmp_item(a,b):
    if int(a.split("_")[-1])>int(b.split("_")[-1]):
        return 1
    elif int(a.split("_")[-1])==int(b.split("_")[-1]):
        return 0
    else:
        return -1

from functools import cmp_to_key
cmp_key = cmp_to_key(cmp_item)

from random import shuffle
def main():
    global S_D
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))


    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")


    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')

    train_dataset= h5py.File("combined-all.h5","r")
    keys= list(train_dataset.keys())
    keys.sort(key=cmp_key)

    for iteration_num in range(0,5):
        # dataset = h5py.File(args.dataset, 'r')
        # train_dataset= h5py.File("combined-all.h5","r")
        reward_dict={}

        # num_videos = len(dataset.keys())
        num_videos= len(train_dataset.keys())
        splits = read_json(args.split)
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        split = splits[args.split_id]
        # train_keys = split['train_keys']
        if iteration_num ==0:
            train_keys= keys[5:-1]
            train_keys.append(keys[-1])
            test_keys= keys[0:5]
        else:
            train_keys=keys[0:(iteration_num*5)]+keys[((iteration_num+1)*5):-1]
            train_keys.append(keys[-1])
            test_keys= keys[(iteration_num*5):((iteration_num+1)*5)]
        shuffle(train_keys)
        # print(test_keys)
        # train_keys= list(train_dataset.keys())
        # test_keys = split['test_keys']
        print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

        print("Initialize model")
        # model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
        # num_networks=3
        num_networks=args.num_networks
        print("Number of networks {}".format(str(num_networks)))
        model= ensemble_DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell, num_networks=args.num_networks)
        print(model)
        # adversary= make_adversary(1518, device)
        # my-change
        # model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout=dropout)
        # print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
        optimizers=[]
        schedulers=[]
        for i in range(num_networks):

            optimizers.append(torch.optim.Adam(model.arr[i].parameters(), lr=args.lr, weight_decay=args.weight_decay))
        if args.stepsize > 0:
            for i in range(num_networks):
                schedulers.append( lr_scheduler.StepLR(optimizers[i], step_size=args.stepsize, gamma=args.gamma))

        if args.resume:
            print("Loading checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            start_epoch = 0

        if use_gpu:
            # model = nn.DataParallel(model).cuda()
            S_D= S_D.to(device)
            model=model.to(device)

        if args.evaluate:
            print("Evaluate only")
            evaluate(model, dataset, test_keys, use_gpu)
            return

        print("==> Start training "+str(iteration_num))
        start_time = time.time()
        model.train()
        baselines = {key: 0. for key in train_keys} # baseline rewards for videos
        reward_writers = {key: [] for key in train_keys} # record reward changes for each video
        reward_writers_nll = {key: [] for key in train_keys}
        activation = nn.Softmax(dim=1)
        epis_reward_nll=[]
        # loss = nn.NLLLoss()
        loss= nn.CrossEntropyLoss()
        label= torch.full((1,), 1, dtype=torch.long).to(device)
        save_seq=None
        count=0
        max_mean=0

        # for epoch in range(start_epoch, args.max_epoch):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs) # shuffle indices
        print((len(idxs)/num_networks))
        # split_idxs=np.zeros((num_networks,int(len(idxs)/num_networks)))
        split_idxs=[]
        for x in range(num_networks):
            if x ==num_networks-1:
                in_arr=list(idxs[x*int(len(idxs)/num_networks):-1])
                in_arr.append(idxs[-1])
                split_idxs.append(in_arr)
            else:
                in_arr=list(idxs[x*int(len(idxs)/num_networks):(x+1)*int(len(idxs)/num_networks)])
                split_idxs.append(in_arr)
        classifier= make_classifier(num_networks)
        classifier= classifier.to(device)
        keys_map={}
        for t in range(len(split_idxs)):
            for z in split_idxs[t]:
                keys_map[train_keys[z]]=t

        classifier= learn_distribution(classifier,train_keys,train_dataset,keys_map,args)
        for epoch in range(start_epoch, 60):
            for i in range(num_networks):
                np.random.shuffle(split_idxs[i])

            for network in range(num_networks):
                indexes = split_idxs[network]
                for idx in indexes:
                    key=train_keys[idx]
                    # key = train_keys[idx]

                    # seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)

                    seq = train_dataset[key]['features'][...] # sequence of features, (seq_len, dim)

                    seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)

                    if use_gpu: seq = seq.cuda()
                    probs = model(seq, network) # output shape (1, seq_len, 1)


                    cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
                    m = Bernoulli(probs)
                    epis_rewards = []
                    for _ in range(args.num_episode):
                        actions = m.sample()
                        log_probs = m.log_prob(actions)

                        if count==0:

                            pick_idxs = actions.squeeze().nonzero().squeeze()
                            save_seq=seq[:,pick_idxs,:]
                            count=1

                        reward, nll = compute_reward(seq, actions,S_D, loss=loss, label=label, activation=activation ,use_gpu=use_gpu, device=device, save_seq=save_seq)

                        expected_reward = log_probs.mean() * (reward - baselines[key])
                        try:
                            r= reward_dict[key]
                            if r<expected_reward:
                                reward_dict[key]=expected_reward.detach()
                        except Exception as e:
                            reward_dict[key]=expected_reward
                        reward_dict[key]= expected_reward.detach()
                        cost -= expected_reward # minimize negative expected reward
                        epis_rewards.append(reward.item())
                        epis_reward_nll.append(nll.item())

                    optimizers[network].zero_grad()
                    cost.backward()
                    torch.nn.utils.clip_grad_norm_(model.arr[network].parameters(), 5.0)
                    optimizers[network].step()
                    baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
                    reward_writers[key].append(np.mean(epis_rewards))
                    reward_writers_nll[key].append(np.mean(epis_reward_nll))

                    # mean=get_f_mean(model, dataset, test_keys, use_gpu, i=i)
                    # if mean > max_mean:
                        # max_mean=mean
            if (epoch)%5==0:
                evaluate_save(model, dataset, test_keys, use_gpu, i=iteration_num, num_networks=args.num_networks, classifier=classifier)
                for n in range(args.num_networks):
                    f_mean=evaluate_network_save(model, dataset, test_keys, use_gpu, i=iteration_num, num_networks=args.num_networks, test_network=n)
                    print_save("Network {} F_MEAN: {}".format(str(n),str(f_mean)))

            epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
            epoch_nll_reward= np.mean([reward_writers_nll[key][epoch] for key in train_keys])
            print("epoch {}/{}\t reward {}\t nll_reward {}\t".format(epoch+1, args.max_epoch, epoch_reward, epoch_nll_reward))


        write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
        # evaluate(model, dataset, test_keys, use_gpu)
        evaluate_save(model, dataset, test_keys, use_gpu, i=iteration_num, num_networks=args.num_networks,classifier=classifier)
        for n in range(args.num_networks):
            f_mean=evaluate_network_save(model, dataset, test_keys, use_gpu, i=iteration_num, num_networks=args.num_networks, test_network=n)
            print_save("Network {} F_MEAN: {}".format(str(n),str(f_mean)))
        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        # model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
        # model_state_dict = model.state_dict() if use_gpu else model.state_dict()
        # model_save_path = osp.join(args.save_dir, 'model_epoch_'+str(i)+"_" + str(args.max_epoch) + '.pth.tar')
        # save_checkpoint(model_state_dict, model_save_path)
        # print("Model saved to {}".format(model_save_path))
    dataset.close()

def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

def print_save(txt, location="results.txt"):
    f = open(location,"a")
    f.write(str(txt)+" \n")
    f.close()
def get_f_mean(model, dataset, test_keys, use_gpu, i=1000):
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)


    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    model.train()
    return mean_fm

def evaluate_network_save(model, dataset, test_keys, use_gpu, i=1000, num_networks=3, test_network=0, classifier=None):
    import math
    if num_networks%2==0:
        majority_vote= (num_networks/2)+1
    else:
        majority_vote=math.ceil(num_networks/2)
    # majority_vote=2
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq,0, eval=True, eval_network=test_network)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    model.train()
    return mean_fm


def evaluate_save(model, dataset, test_keys, use_gpu, i=1000, num_networks=3, classifier= None):
    print_save("==> Test")
    if num_networks%2==0:
        majority_vote= (num_networks/2)+1
    else:
        majority_vote=math.ceil(num_networks/2)
    # majority_vote=2

    print(test_keys)
    print(dataset.keys())
    list_priority_networks=[]
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):

            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)

            if use_gpu: seq = seq.cuda()
            if classifier is not None:
                print(nn.functional.sigmoid(classifier(seq)).squeeze(0))
                priority_network=torch.argmax(nn.functional.sigmoid(classifier(seq)).squeeze(0))
                list_priority_networks.append(priority_network.detach().cpu().numpy())
            machine_summaries=[]
            for n in range(num_networks):
                probs = model(seq,n, eval=True, eval_network=n)
                probs = probs.data.cpu().squeeze().numpy()

                cps = dataset[key]['change_points'][...]
                num_frames = dataset[key]['n_frames'][()]
                nfps = dataset[key]['n_frame_per_seg'][...].tolist()
                positions = dataset[key]['picks'][...]
                user_summary = dataset[key]['user_summary'][...]
                if n==0:

                    machine_summaries=vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)


                else:
                    machine_summaries=machine_summaries+vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
                if priority_network is not None and priority_network==n:
                    machine_summaries=machine_summaries+1
            machine_summaries[machine_summaries<majority_vote]=0
            machine_summaries[machine_summaries>=majority_vote]=1

            machine_summary=machine_summaries


            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print_save(tabulate(table))
        print_save(list_priority_networks)

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print_save("Average F-score {:.1%}".format(mean_fm))
    print_save("Iteration Number: "+str(i))
    model.train()
    return mean_fm

if __name__ == '__main__':
    main()
