
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

# Only imports so far
import torch
import torch.nn as nn
from torch.nn import functional as F
from attribution import softmax
import os
import numpy as np

# make directory given its path
def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# Layer Norm Implementation (just a torch.nn.Module)
class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']


    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = args.lnv # this lnv was set to 'nowb'
        self.sigma = args.sigma # None
        self.hidden = hidden # None
        self.adanorm_scale = args.adanorm_scale # 1
        self.nowb_scale = args.nowb_scale # None
        self.mean_detach = args.mean_detach # False
        self.std_detach = args.std_detach # False
        self.gamma_LN = args.gamma_LN if hasattr(args, "gamma_LN") else 0.0

        # prolly never go into this case, meaning elementwise_affine stays True
        if self.mode == 'no_norm':
            elementwise_affine = False
        
        self.eps = eps # 1e-5
        self.elementwise_affine = elementwise_affine
        
        # if elementwise_affine given, create tensors of size hidden
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))

        # else do this, ie, no weight and bias introduced, hence no affine transformations
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        
        self.reset_parameters() # reinit w,b if elementwise_affine is True

    # used at the end of init
    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)


    def forward(self, input):

        # Skip normalization
        # do not perform normalization, just an identity connection 
        if self.mode == 'no_norm':
            return input
        

        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T*B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps)
        

        elif self.mode == 'adanorm':
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm*self.adanorm_scale
        
        # performs centering and scaling but does not apply the learned affine transformations
        elif self.mode == 'nowb':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            # get mean, standard_deviation 

            # mean is now separate from the computation graph
            # if self.mean_detach:
            #     mean = mean.detach()
        
            # # std is now separate from the computation graph
            # if self.std_detach:
            #     std = std.detach()

            # the above two segments make LayerNorm locally linear wrt x (ie, (x - const1)/const2)

            # implementing the gamma GI flow
            gamma = self.gamma_LN
            print("DEBUG: gamma_LN inside LN =", self.gamma_LN)
            mean = (1 - gamma)*mean.detach() +  gamma*mean
            std = (1 - gamma)*std.detach() +  gamma*std
            
            input_norm = (input - mean) / (std + self.eps)
            return input_norm

        elif self.mode == 'distillnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)

            input_norm = input_norm*self.weight + self.bias

            return input_norm

        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None):
    # args not None
    if args is not None:
        # if args is not None, return the custom LN implementation
        if args.lnv != 'origin':
            return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    
    # this just gets LN 
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

# evaluation helper (perturbation utility)
def flip(model, x, token_ids, tokens, y_true,  fracs, flip_case,random_order = False, tokenizer=None, device='cpu'):
    """
    x - the relevance scores
    token_ids - model inputs
    tokens - strings (for logging)
    fracs - sequence of fractions at which to evaluate
    flip_case - [generate, pruning] - standard perturbation modes
    """
    x = np.array(x)

    # get original logits
    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)
    y0 = model(inputs0, labels = None)['logits'].squeeze().detach().cpu().numpy()
    orig_token_ids = np.copy(token_ids.detach().cpu().numpy())

    # 
    if random_order==False:
        # highest relevance added first
        if  flip_case=='generate':
            inds_sorted = np.argsort(x)[::-1]
        
        # least relevant removed first
        elif flip_case=='pruning':
            inds_sorted =  np.argsort(np.abs(x))
        
        else:
            raise
    
    # random shuffling
    else:
        inds_ = np.array(list(range(x.shape[-1])))
        remain_inds = np.array(inds_)
        np.random.shuffle(remain_inds)

        inds_sorted = remain_inds


    inds_sorted = inds_sorted.copy()
    vals = x[inds_sorted]

    mse = []
    evidence = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().cpu().numpy(), 'y0':y0}

    N=len(x)

    evolution = {}

    # for each fraction, 
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        # set selected token to UNK
        if flip_case == 'pruning':
            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

        # start from UNK and set selected token positions to original token ids, preserve pad
        elif flip_case == 'generate':
            inputs = UNK_IDX*torch.ones_like(inputs0)
            # Set pad tokens
            inputs[inputs0==0] = 0

            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]

        # run model on modified inputs and get new logits
        y = model(inputs, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().numpy()
        y = y.squeeze()

        err = np.sum((y0-y)**2)
        mse.append(err)
        evidence.append(softmax(y)[int(y_true)])

      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:
        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
    return mse, evidence, model_outs
