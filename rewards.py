import torch
import sys
import numpy as np

def complete_summary(sd, ntokens=64, device=None, save_seq= None):
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

def compute_reward(seq, actions, disc, loss=None, label=None, activation=None,ignore_far_sim=True,temp_dist_thre=20, use_gpu=False, device=None, save_seq=None):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    # _seq = seq.detach()
    _seq= seq
    # _actions = actions.detach()
    _actions=actions
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    if num_picks == 0 or len(_seq[:,pick_idxs,:].shape)<3:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward,reward
    summary= complete_summary(_seq[:,pick_idxs,:],device=device, save_seq=save_seq)

    # transformer model will be passed from main function.
    src_mask= disc.generate_square_subsequent_mask(seq.size(0))
    # print(seq.size())
    output = disc(summary,src_mask)
    # print(activation(output))
    # hj()
    nll= loss(output, label)



    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    # reward=nll
    reward = (((reward_div + reward_rep) * 0.5))*0.3+ (0.7*nll)
    # reward=0*((reward_div + reward_rep) * 0.5)
    # reward= (reward_div + reward_rep) * 0.5

    return reward, nll

def compute_reward_old(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    reward_rep = torch.exp(-dist_mat.mean())

    # combine the two rewards
    reward = (reward_div + reward_rep) * 0.5

    return reward
