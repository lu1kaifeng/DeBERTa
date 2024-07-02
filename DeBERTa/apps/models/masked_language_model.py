#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#
""" Masked Language model for representation learning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor

import csv
import os
import json
import random
import time
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import pdb
from collections.abc import Mapping
from copy import copy

from msg3d.model.ms_gcn import MultiScale_GraphConv
from ...deberta import *

__all__ = ['MaskedLanguageModel']




class EnhancedMaskDecoder(torch.nn.Module):
  def __init__(self, config, vocab_size):
    super().__init__()
    self.config = config
    self.position_biased_input = getattr(config, 'position_biased_input', True)
    self.lm_head = BertLMPredictionHead(config, vocab_size)
    self.graph = getattr(config, 'graph', False)
    self.graph_conv_enable = getattr(config, 'graph_conv', False)
    if self.graph_conv_enable:
      self.graph_conv = MultiScale_GraphConv(config.graph_gcn_scales, config.graph_hidden_size, config.hidden_size)
    else:
      self.graph_conv = None
  def forward(self, ctx_layers, ebd_weight,edge_ebd_weight, target_ids, input_ids, input_mask, z_states, attention_mask, encoder, relative_pos=None,adj_mat=None,adj_label=None):
    if self.graph_conv_enable:
      graph_embed = self.graph_conv((adj_mat['lookup'] > 0).to(torch.float32),adj_mat['A_powers'],adj_mat['lookup'],adj_mat['last_edge'],adj_mat['edge_embedding'])
    else:
      graph_embed = None
    mlm_ctx_layers,unflattened = self.emd_context_layer(ctx_layers, z_states, attention_mask, encoder, target_ids, input_ids, input_mask, relative_pos=relative_pos,adj_mat=adj_mat,graph_embed =graph_embed)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    lm_loss = torch.tensor(0).to(ctx_layers[-1])
    arlm_loss = torch.tensor(0).to(ctx_layers[-1])
    ctx_layer = mlm_ctx_layers[-1]
    unflattened_ctx_layer = unflattened[-1]
    lm_logits = self.lm_head(ctx_layer, ebd_weight).float()
    lm_logits = lm_logits.view(-1, lm_logits.size(-1))
    gm_loss = 0
    gm_logits = None
    if self.graph:
      edge_labels = adj_label
      non_zero = edge_labels.nonzero()
      if not self.graph_conv_enable:
        gm_logits = self.lm_head.edge_forward(unflattened_ctx_layer,non_zero,edge_ebd_weight)
      else:
        gm_logits = self.lm_head.edge_forward(unflattened_ctx_layer, non_zero, edge_ebd_weight)
      edge_labels_labels = edge_labels[non_zero[:, 0], non_zero[:, 1], non_zero[:, 2]] - 1
      gm_loss = loss_fct(gm_logits,edge_labels_labels.long())
      #lasso = 0.001 * torch.norm(self.lm_head.dense.weight, 1)
      ridge = 0.001 * torch.norm(self.lm_head.dense.weight, 2)
      gm_loss+=ridge
      #gm_loss += lasso
      pass
    lm_labels = target_ids.view(-1)
    label_index = (target_ids.view(-1)>0).nonzero().view(-1)
    lm_labels = lm_labels.index_select(0, label_index)
    lm_loss = loss_fct(lm_logits, lm_labels.long())
    lm_loss *= 0.1
    return lm_logits, lm_labels, lm_loss,gm_logits,gm_loss,edge_labels_labels

  def emd_context_layer(self, encoder_layers, z_states, attention_mask, encoder, target_ids, input_ids, input_mask, relative_pos=None,adj_mat=None,graph_embed = None):
    if attention_mask.dim()<=2:
      extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      att_mask = extended_attention_mask.byte()
      attention_mask = att_mask*att_mask.squeeze(-2).unsqueeze(-1)
    elif attention_mask.dim()==3:
      attention_mask = attention_mask.unsqueeze(1)
    target_mask = target_ids>0
    hidden_states = encoder_layers[-2]
    if not self.position_biased_input: 
      layers = [encoder.layer[-1] for _ in range(2)]
      z_states +=  hidden_states
      query_states = z_states
      query_mask = attention_mask
      outputs = []
      rel_embeddings = encoder.get_rel_embedding()

      for layer in layers:
        # TODO: pass relative pos ids
        output = layer(hidden_states, query_mask, return_att=False, query_states = query_states, relative_pos=relative_pos, rel_embeddings = rel_embeddings,adj_mat=adj_mat,graph_embed=graph_embed)
        query_states = output
        outputs.append(query_states)
    else:
      outputs = [encoder_layers[-1]]
    
    _mask_index = (target_ids>0).view(-1).nonzero().view(-1)
    def flatten_states(q_states):
      q_states = q_states.view((-1, q_states.size(-1)))
      q_states = q_states.index_select(0, _mask_index)
      return q_states

    return [flatten_states(q) for q in outputs],outputs

class MaskedLanguageModel(NNModule):
  """ Masked language model with DeBERTa
  """
  def __init__(self, config, *wargs, **kwargs):
    super().__init__(config)
    self.deberta = DeBERTa(config)

    self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
    self.position_buckets = getattr(config, 'position_buckets', -1)
    if self.max_relative_positions <1:
      self.max_relative_positions = config.max_position_embeddings
    self.lm_predictions = EnhancedMaskDecoder(self.deberta.config, self.deberta.embeddings.word_embeddings.weight.size(0))
    self.apply(self.init_weights)

  def forward(self, input_ids, input_mask=None, labels=None, position_ids=None, attention_mask=None, adj_mat = None,adj_label=None):
    device = list(self.parameters())[0].device
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    type_ids = None
    lm_labels = labels.to(device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(device)
    else:
      attention_mask = input_mask

    encoder_output = self.deberta(input_ids, input_mask, type_ids, output_all_encoded_layers=True, position_ids = position_ids,adj_mat=adj_mat)
    encoder_layers = encoder_output['hidden_states']
    z_states = encoder_output['position_embeddings']
    ctx_layer = encoder_layers[-1]
    lm_loss = torch.tensor(0).to(ctx_layer).float()
    lm_logits = None
    label_inputs = None
    if lm_labels is not None:
      ebd_weight = self.deberta.embeddings.word_embeddings.weight
      edge_ebd_weight = self.deberta.embeddings.edge_embeddings.weight

      label_index = (lm_labels.view(-1) > 0).nonzero()
      label_inputs = torch.gather(input_ids.view(-1), 0, label_index.view(-1))
      if label_index.size(0)>0:
        (lm_logits, lm_labels, lm_loss,gm_logits,gm_loss,gm_labels) = self.lm_predictions(encoder_layers, ebd_weight,edge_ebd_weight, lm_labels, input_ids, input_mask, z_states, attention_mask, self.deberta.encoder,adj_mat=encoder_output['adj_embeddings'],adj_label=adj_label)

    return {
            'logits' : lm_logits,
            'labels' : lm_labels,
            'loss' : lm_loss.float(),
            'gm_loss': gm_loss.float(),
            'gm_logits': gm_logits,
      'gm_labels': gm_labels,
          }
