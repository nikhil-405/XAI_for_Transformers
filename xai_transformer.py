# Imports
import numpy as np
from torch import nn
import torch
import sys
import copy
root_dir = './../'
sys.path.append(root_dir)
from utils import LayerNorm # this is used for the custom LayerNormImp class

# LN arguments for various forms of testing
class LNargs(object):
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False

# this is the config for LRP-LN rule within the paper
class LNargsDetach(object):
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        # these parameters are now True
        self.mean_detach = True
        self.std_detach = True

# Detaches std but not mean
class LNargsDetachNotMean(object):
    def __init__(self):
        self.lnv = 'nowb'
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True

# ----------------------------------------------------------------------------
# does not seem relevant

def make_p_layer(layer, gamma):
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight+gamma*layer.weight.clamp(min=0))
    player.bias   = torch.nn.Parameter(layer.bias +gamma*layer.bias.clamp(min=0))
    return player

# ----------------------------------------------------------------------------

# Learnable layer wise gammas
class GammaParams(torch.nn.Module):
    def __init__(self, n_blocks, init_val = 0.0, device = None):
        super().__init__()
        raw_init = float(np.log(init_val + 1e-6) - np.log(1.0 - init_val + 1e-6)) if init_val > 0 else 0.0

        self.raw_AH = torch.nn.Parameter(torch.full((n_blocks,), raw_init, dtype = torch.float))
        self.raw_LN = torch.nn.Parameter(torch.full((n_blocks,), raw_init, dtype = torch.float))

        if device is not None:
            self.to(device)


    def forward(self):
        return torch.sigmoid(self.raw_AH), torch.sigmoid(self.raw_LN)

# Small MLP
# returns per layer gammas
class GammaNet(nn.Module):
    def __init__(self, hidden_dim, n_blocks, pooled_dim = 768):
        super().__init__()
        self.n_blocks = n_blocks
        self.net = nn.Sequential(
                        nn.Linear(pooled_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 2*n_blocks), # want to return for LN, AH
                        nn.Sigmoid()
                    )

    def forward(self, x):
        # x: pooled
        out = self.net(x)
        return out.view(-1, 2, self.n_blocks)


# Bert's pooling layer
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. ## --> this is often times the [CLS] token
        self.first_token_tensor = hidden_states[:, 0]
        self.pooled_output1 = self.dense(self.first_token_tensor)
        self.pooled_output2 = self.activation(self.pooled_output1)
        return self.pooled_output2


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

        if self.config.train_mode == True:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        # choose which LN to use based on the config
        if config.detach_layernorm == True:
            assert config.train_mode==False

            if config.detach_mean==False:
                print('Detach LayerNorm only Norm')
                largs = LNargsDetachNotMean()
            else:
                print('Detach LayerNorm Mean+Norm')
                largs = LNargsDetach()
        else:
            largs =  LNargs()

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps, args=largs)

        self.detach = config.detach_layernorm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)

        if self.config.train_mode == True:
            hidden_states = self.dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    # this function is used within the explanation mode
    def pforward(self, hidden_states, input_tensor, gamma_LN):
        pdense =  make_p_layer(self.dense, gamma_LN)
        hidden_states = pdense(hidden_states)
        #hidden_states = self.dense(hidden_states)
        if self.config.train_mode == True:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor, gamma_LN_override = gamma_LN)

        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Q, K, V
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)

        self.output = BertSelfOutput(config)
        self.detach = config.detach_kq

        if self.config.train_mode == True:
            self.dropout =  torch.nn.Dropout(p=0.1, inplace=False)

        if self.detach == True:
            assert self.config.train_mode==False
            print('Detach K-Q-softmax branch')


    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, 768])
        # xout torch.Size([1, 10, 12, 64])
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, self.config.attention_head_size)
        x = x.view(*new_x_shape)
        X= x.permute(0, 2, 1, 3)
        return X

    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0,  1, 2, 3)
        return x.reshape(old_shape)

    @staticmethod
    def pproc(layer,player,x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data


    def process_block(self, hidden_states, gamma_AH=0.0, gamma_LN=0.0, method=None, debug=False):
      # ============ FIX: Separate scalar vs broadcasted gamma ============
        # Ensure gamma_AH is a scalar tensor with grad
        if not isinstance(gamma_AH, torch.Tensor):
            gamma_AH = torch.tensor(float(gamma_AH), device=hidden_states.device, dtype=torch.float32)

        # Scalar version for make_p_layer (keeps weight matrices 2D)
        gamma_AH_scalar = gamma_AH

        # Broadcasted version for attention interpolation
        gamma_AH_broadcast = gamma_AH_scalar.view(1, 1, 1, 1)

        # print(f"    [DEBUG] gamma_AH scalar value: {gamma_AH_scalar.item():.4f}, shape: {gamma_AH_scalar.shape}")
        # print(f"    [DEBUG] gamma_AH broadcast shape: {gamma_AH_broadcast.shape}")

        # Create parameterized layers using SCALAR gamma
        pquery = make_p_layer(self.query, gamma_AH_scalar)
        pkey = make_p_layer(self.key, gamma_AH_scalar)
        pvalue = make_p_layer(self.value, gamma_AH_scalar)

        n_nodes = hidden_states.shape[1]

        # Forward through Q,K,V linear layers
        if self.config.train_mode:
            query_ = self.query(hidden_states)
            key_ = self.key(hidden_states)
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states)
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        # Reshape for multi-head attention
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # Compute attention scores
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))
        attn = nn.functional.softmax(attention_scores, dim=-1)

        # ============ FIX: Use BROADCASTED gamma for interpolation ============
        # This ensures gamma appears in the computation graph correctly
        attention_probs = (1.0 - gamma_AH_broadcast) * attn.detach() + gamma_AH_broadcast * attn

        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Reshape back to [batch, seq_len, hidden_size]
        old_context_layer_shape = context_layer.shape
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output with LayerNorm (pass scalar gamma_LN)
        if self.config.train_mode:
            output = self.output(context_layer, hidden_states)
        else:
            output = self.output.pforward(context_layer, hidden_states, gamma_LN=gamma_LN)

        return output, attention_probs

    def forward(self, hidden_states, gamma_AH=0.0, gamma_LN=0.0, method=None, debug=False):
        return self.process_block(hidden_states, gamma_AH, gamma_LN, method, debug)



class BertAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config.n_blocks
        self.n_blocks=n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(n_blocks)])
        self.output = BertSelfOutput(config)

        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.n_classes, bias=True)
        self.device = config.device

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams      = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}


    def forward_simple(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Simple forward pass without any explanation logic
        """
        # Get embeddings
        hidden_states = self.embeds(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0
        ).to(self.config.device)

        # Forward through attention layers with default gammas
        attn_input = hidden_states
        for block in self.attention_layers:
            attn_input, _ = block(attn_input, gamma_AH=0.0, gamma_LN=0.0)

        # Pool and classify
        pooled = self.pooler(attn_input)
        logits = self.classifier(pooled)

        return {'logits': logits}

    def forward(self, hidden_states, gamma_AH=0.0, gamma_LN=0.0, method=None, debug=False):
        # For GammaNet training, use direct interpolation (no pproc)
        # pproc breaks gradients - only use it for static explanation mode

        query_ = self.query(hidden_states)
        key_ = self.key(hidden_states)
        val_ = self.value(hidden_states)

        # Transpose for attention
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # Attention scores
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))
        attn = nn.functional.softmax(attention_scores, dim=-1)

        # === KEY CHANGE: Differentiable gamma interpolation ===
        # No .detach() on the whole tensor - only on the softmax gate values
        # This preserves gradient flow to gamma_AH
        attention_probs = (1.0 - gamma_AH) * attn.detach() + gamma_AH * attn

        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.config.all_head_size,))

        # Output with LayerNorm gamma interpolation
        if self.config.train_mode:
            output = self.output(context_layer, hidden_states)
        else:
            # Use pforward which handles gamma_LN correctly
            output = self.output.pforward(context_layer, hidden_states, gamma_LN=gamma_LN)

        return output, attention_probs


    def prep_lrp(self, x):
        x = x.data
        x.requires_grad_(True)
        return x


    # V. imp
    def forward_and_explain(self, input_ids,
                                  cl,
                                  attention_mask=None,
                                  token_type_ids=None,
                                  position_ids=None,
                                  inputs_embeds=None,
                                  labels=None,
                                  past_key_values_length=0,
                                  method=None,
                                  gammas = None,
                                  gamma_net = None):


        # Forward
        # dictionary used to store the entire backward pass
        A = {}

        hidden_states= self.embeds(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          position_ids=None,
                                          inputs_embeds=None,
                                          past_key_values_length=0).to(self.config.device)

        A['hidden_states'] = hidden_states

        attn_input = hidden_states


        #################### Code for layer wise learnable gamma here ####################
        n_blocks = len(self.attention_layers)

        # using the CLS embedding hoping that it would have captured global summary
        try:
            pooled_for_gamma = hidden_states[:, 0, :]
        except Exception:
            pooled_for_gamma = hidden_states.mean(dim = 1)

        if (gamma_net is not None):
            gamma_tensor = gamma_net(pooled_for_gamma)
            if (gamma_tensor.dim() == 3):
                gamma_tensor = gamma_tensor[0]

            gamma_AH = gamma_tensor[0]
            gamma_LN = gamma_tensor[1]

        elif gammas is not None:
            gamma_AH, gamma_LN = gammas
            if not torch.is_tensor(gamma_AH):
                gamma_AH = torch.tensor(gamma_AH,
                                        device = hidden_states.device,
                                        dtype = torch.float)

            if not torch.is_tensor(gamma_LN):
                gamma_LN = torch.tensor(gamma_LN,
                                        device = hidden_states.device,
                                        dtype = torch.float)
        else:
            gamma_AH = torch.full((n_blocks,), float(self.config.gamma_AH), device=hidden_states.device, dtype=torch.float)
            gamma_LN = torch.full((n_blocks,), float(self.config.gamma_LN), device=hidden_states.device, dtype=torch.float)
        #################### ####################

        for i,block in enumerate(self.attention_layers):
            if torch.is_tensor(gamma_AH):
                gamma_AH_i = gamma_AH[i].to(hidden_states.device)
            else:
                gamma_AH_i = torch.tensor(float(gamma_AH[i]), device=hidden_states.device, dtype=torch.float)

            if torch.is_tensor(gamma_LN):
                gamma_LN_i = gamma_LN[i].to(hidden_states.device)
            else:
                gamma_LN_i = torch.tensor(float(gamma_LN[i]), device=hidden_states.device, dtype=torch.float)

            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.clone().requires_grad_(True)
            # attn_inputdata.requires_grad_(True)
            attn_inputdata.retain_grad()

            A['attn_input_{}_data'.format(i)] = attn_inputdata
            A['attn_input_{}'.format(i)] = attn_input


          #  print('using gamma', gamma)

            output, attention_probs = block(A['attn_input_{}_data'.format(i)], gamma_AH = gamma_AH_i, gamma_LN = gamma_LN_i, method=method)

            self.attention_probs[i] = attention_probs
            attn_input = output


        # (1, 12, 768) -> (1x768)

        outputdata = output.clone().requires_grad_(True)
        outputdata.retain_grad()

        pooled = self.pooler(outputdata) #A['attn_output'] )

        # (1x768) -> (1,nclasses)
        pooleddata = pooled.clone()
        pooleddata.retain_grad()
        logits = self.classifier(pooleddata)

        A['logits'] = logits

        # Through clf layer
        Rout = A['logits'][:,cl]

        self.R0 = Rout.detach().cpu().numpy()

        Rout.backward()
        ((pooleddata.grad)*pooled).sum().backward()

        Rpool = ((outputdata.grad)*output)

        R_ = Rpool
        for i,block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = A['attn_input_{}_data'.format(i)].grad
            R_attn =  (R_grad)*A['attn_input_{}'.format(i)]
            if method == 'GAE':
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

        R = R_.sum(2)


        if labels is not None:
            loss =  torch.nn.CrossEntropyLoss()(logits,labels)
        else:
            loss = None

        return {'loss': loss, 'logits': logits, 'R': R}

    def forward_and_explain_differentiable(self, input_ids, cl, gamma_net=None, gammas=None):
      """
      DEBUG VERSION: Returns gamma values for regularization
      """
      # print("\n" + "="*60)
      # print("FORWARD PASS START")
      # print("="*60)

      # Forward embeddings
      hidden_states = self.embeds(input_ids).to(self.config.device)
      hidden_states.requires_grad_(True)
      embeddings = hidden_states
      # print(f"[1] Embeddings shape: {embeddings.shape}, requires_grad: {embeddings.requires_grad}")

      # ============ GET GAMMAS ============
      if gamma_net is not None:
          pooled_for_gamma = embeddings[:, 0, :]
          # print(f"[2] Pooled for GammaNet shape: {pooled_for_gamma.shape}")

          gamma_tensor = gamma_net(pooled_for_gamma)
          # print(f"[3] GammaNet output shape: {gamma_tensor.shape}")
          # print(f"[3a] GammaNet output requires_grad: {gamma_tensor.requires_grad}")
          # print(f"[3b] GammaNet output grad_fn: {gamma_tensor.grad_fn}")

          # Add gradient hook
          def gamma_hook(grad):
              # print(f"[HOOK] Gradient reaching GammaNet output: shape={grad.shape}, norm={grad.norm().item():.6f}")
              return grad
          gamma_tensor.register_hook(gamma_hook)

          gamma_AH = gamma_tensor[0, 0, :]  # Shape: [n_blocks]
          gamma_LN = gamma_tensor[0, 1, :]  # Shape: [n_blocks]
          # print(f"[4] gamma_AH shape: {gamma_AH.shape}, requires_grad: {gamma_AH.requires_grad}")
          # print(f"[5] gamma_LN shape: {gamma_LN.shape}, requires_grad: {gamma_LN.requires_grad}")
      else:
          gamma_AH, gamma_LN = gammas

      # ============ FORWARD PASS ============
      attn_input = hidden_states

      for i, block in enumerate(self.attention_layers):
        #   print(f"\n--- Block {i} ---")

          # Get per-layer gamma values
          if torch.is_tensor(gamma_AH):
              gamma_AH_i = gamma_AH[i]  # Scalar tensor
              gamma_LN_i = gamma_LN[i]
          else:
              gamma_AH_i = torch.tensor(float(gamma_AH[i]), device=hidden_states.device)
              gamma_LN_i = torch.tensor(float(gamma_LN[i]), device=hidden_states.device)

          # print(f"[7.{i}] gamma_AH_i: {gamma_AH_i.item():.4f}, requires_grad: {gamma_AH_i.requires_grad}")
          # print(f"[8.{i}] gamma_LN_i: {gamma_LN_i.item():.4f}, requires_grad: {gamma_LN_i.requires_grad}")

          # Forward through block
          # Pass the scalar gamma values - block will handle broadcasting
          output, _ = block.process_block(attn_input, gamma_AH=gamma_AH_i, gamma_LN=gamma_LN_i)
          attn_input = output

          # print(f"[9.{i}] output shape: {output.shape}, requires_grad: {output.requires_grad}")

      # Final layers
      # print("\n--- Final Layers ---")
      pooled = self.pooler(attn_input)
      logits = self.classifier(pooled)

      Rout = logits[:, cl]
      # print(f"[12] Rout shape: {Rout.shape}, requires_grad: {Rout.requires_grad}")

      # Compute R
      R_grad = torch.autograd.grad(Rout, embeddings,
                                  grad_outputs=torch.ones_like(Rout),
                                  create_graph=True)[0]
      R = (R_grad * embeddings).sum(dim=-1)
      # print(f"[14] Final R shape: {R.shape}, requires_grad: {R.requires_grad}")
      # print("\n" + "="*60 + "\n")

      # ============ RETURN EVERYTHING ============
      return {
          'loss': None,
          'logits': logits,
          'R': R,
          'gamma_AH': gamma_AH,  # Return these for regularization!
          'gamma_LN': gamma_LN
      }