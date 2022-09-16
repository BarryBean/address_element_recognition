# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.crf import CRF
from model.layers import NERmodel
from transformers.models.bert.modeling_bert import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GazLSTM(nn.Module):
    def __init__(self, data):
        super(GazLSTM, self).__init__()

        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        self.use_char = data.HP_use_char
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.use_count = data.HP_use_count
        self.num_layer = data.HP_num_layer
        self.model_type = data.model_type
        self.use_bert = data.use_bert
        self.type_length = data.type_alphabet_size + 2
        self.bound_length = data.bound_alphabet_size + 2

        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        data.pretrain_gaz_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])

        if self.use_char:
            scale = np.sqrt(3.0 / self.word_emb_dim)
            data.pretrain_word_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])

        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.biword_alphabet.size(), self.biword_emb_dim)

        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))

        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.word_emb_dim)))
        if self.use_biword:
            if data.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
            else:
                self.biword_embedding.weight.data.copy_(
                    torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), self.word_emb_dim)))

        char_feature_dim = self.word_emb_dim + 4 * self.gaz_emb_dim
        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768

        ## lstm model
        if self.model_type == 'lstm':
            lstm_hidden = self.hidden_dim
            if self.bilstm_flag:
                self.hidden_dim *= 2
            self.NERmodel = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden,
                                     num_layer=self.lstm_layer, biflag=self.bilstm_flag)
            self.bound_model = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden,
                                        num_layer=self.lstm_layer, biflag=self.bilstm_flag)
            self.type_model = NERmodel(model_type='lstm', input_dim=char_feature_dim, hidden_dim=lstm_hidden,
                                       num_layer=self.lstm_layer, biflag=self.bilstm_flag)

        ## cnn model
        if self.model_type == 'cnn':
            self.NERmodel = NERmodel(model_type='cnn', input_dim=char_feature_dim, hidden_dim=self.hidden_dim,
                                     num_layer=self.num_layer, dropout=data.HP_dropout, gpu=self.gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.NERmodel = NERmodel(model_type='transformer', input_dim=char_feature_dim, hidden_dim=self.hidden_dim,
                                     num_layer=self.num_layer, dropout=data.HP_dropout)

        self.drop = nn.Dropout(p=data.HP_dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.use_bert:
            self.bert_encoder = BertModel.from_pretrained('bert-base-chinese')
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        self.T_block1 = I_S_Block(self.hidden_dim)
        self.T_block2 = I_S_Block(self.hidden_dim)

        self.type_fc = nn.Linear(self.hidden_dim, data.type_alphabet_size + 2)
        self.I_S_Emb = Label_Attention(self.hidden2tag, self.hidden2tag)
        self.bound_fc = nn.Linear(self.hidden_dim, data.bound_alphabet_size + 2)

        self.I_S_Emb2 = Label_Attention2(self.bound_fc, self.type_fc, self.hidden2tag)

        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

        self.bound_gru_layer = nn.GRU(char_feature_dim, self.hidden_dim // 2, num_layers=1, batch_first=True,
                                      bidirectional=True)

        self.fusion_layer = FusionLayer(self.hidden_dim, self.hidden_dim, data.HP_dropout, data.label_alphabet_size + 2)
        self.act = nn.Tanh()
        self.word_word_weight = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.last_out = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        attn_W = torch.zeros(self.hidden_dim, self.hidden_dim)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=0.02)
        self.fuse_layernorm = LayerNorm(self.hidden_dim, eps=1e-12)

        self.linear_layer = nn.Linear(self.hidden_dim, data.label_alphabet_size + 2)

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()
            self.NERmodel = self.NERmodel.cuda()
            self.bound_model = self.bound_model.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()
            self.T_block1 = self.T_block1.cuda()
            self.T_block2 = self.T_block2.cuda()
            self.type_fc = self.type_fc.cuda()
            self.I_S_Emb = self.I_S_Emb.cuda()
            self.I_S_Emb2 = self.I_S_Emb2.cuda()
            self.bound_fc = self.bound_fc.cuda()
            self.weight1.data = self.weight1.cuda()
            self.weight2.data = self.weight2.cuda()
            self.weight3.data = self.weight3.cuda()
            self.bound_gru_layer = self.bound_gru_layer.cuda()
            self.fusion_layer = self.fusion_layer.cuda()
            self.type_model = self.type_model.cuda()
            self.act = self.act.cuda()
            self.word_word_weight = self.word_word_weight.cuda()
            self.attn_W.data = self.attn_W.cuda()
            self.fuse_layernorm = self.fuse_layernorm.cuda()
            self.last_out = self.last_out.cuda()
            self.linear_layer = self.linear_layer.cuda()
            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()

    def get_tags(self, gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input,
                 gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask, trans_matrix):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []

        word_embs = self.word_embedding(word_inputs)

        if self.use_biword:
            biword_embs = self.biword_embedding(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs], dim=-1)

        if self.model_type != 'transformer':
            word_inputs_d = self.drop(word_embs)  # (b,l,we)
        else:
            word_inputs_d = word_embs

        if self.use_char:
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1, 1, 1, 1, 1, self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data.bool(), 0)  # (b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  # (b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  # (b,l,4,gl,ce)

            if self.model_type != 'transformer':
                gaz_embeds = self.drop(gaz_embeds)
            else:
                gaz_embeds = gaz_embeds

        else:  # use gaz embedding
            gaz_embeds = self.gaz_embedding(layer_gaz)

            if self.model_type != 'transformer':
                gaz_embeds_d = self.drop(gaz_embeds)
            else:
                gaz_embeds_d = gaz_embeds

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1, 1, 1, 1, self.gaz_emb_dim)

            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data.bool(), 0)  # (b,l,4,g,ge)  ge:gaz_embed_dim

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  # (b,l,4,gn)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  # (b,l,1,1)

            weights = gaz_count.div(count_sum)  # (b,l,4,g)
            weights = weights * 4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights * gaz_embeds  # (b,l,4,g,e)
            gaz_embeds = torch.sum(gaz_embeds, dim=3)  # (b,l,4,e)

        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  # (b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  # (b,l,4,ge)/(b,l,4,1)

        gaz_embeds_cat = gaz_embeds.view(batch_size, seq_len, -1)  # (b,l,4*ge)

        word_input_cat = torch.cat([word_inputs_d, gaz_embeds_cat], dim=-1)  # (b,l,we+4*ge)

        ### cat bert feature
        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long()
            # seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            outputs = outputs[0][:, 1:-1, :]
            word_input_cat = torch.cat([word_input_cat, outputs], dim=-1)

        feature_out_d = self.NERmodel(word_input_cat)

        #### bound
        # lstm
        # bound_out = self.bound_model(word_input_cat)
        # bound_feat = self.bound_fc(bound_out)
        # trans_bert_feats = torch.matmul(bound_feat, trans_matrix.float())

        # gru
        sorted, _ = torch.sort(word_seq_lengths, descending=True)
        packed_heads = pack_padded_sequence(word_input_cat, sorted.cpu(), True)
        hidden = None
        heads_out, hidden = self.bound_gru_layer(packed_heads, hidden)
        heads_out, _ = pad_packed_sequence(heads_out)  # heads_out (seq_len, batch_size, hidden_size)
        heads_feature = self.drop(heads_out.transpose(1, 0))  # heads_out (batch_size, seq_len, hidden_size)
        # heads_outputs = self.bound_fc(heads_feature)
        # trans_bert_feats = torch.matmul(heads_outputs, trans_matrix.float())

        ### type
        # lstm
        # type_out = self.type_model(word_input_cat)

        # 同步塞 bound和type
        H_bound, H_type, H_slot = self.I_S_Emb2(feature_out_d, feature_out_d, feature_out_d)
        H_T, H_S_T = self.T_block1(H_type + feature_out_d, H_slot + feature_out_d, mask)
        H_B, H_S_B = self.T_block2(H_bound + feature_out_d, H_slot + feature_out_d, mask)

        # 全套bca太慢了
        H_ner_t, H_ner_s = self.I_S_Emb(H_S_T, H_S_B)
        H_S1, H_S2 = self.T_block2(H_S_T + H_ner_t, H_S_B + H_ner_s, mask)

        out = self.fusion_layer(H_S1, H_S2)

        tags = self.hidden2tag(feature_out_d)
        # slots = self.hidden2tag(H_S1)
        # slots2 = self.hidden2tag(H_S2)
        types = self.type_fc(H_T)
        heads_outputs = self.bound_fc(heads_feature)
        trans_bert_feats = torch.matmul(heads_outputs, trans_matrix.float())

        # 此时边界和类型的隐藏信息都加进slot中，开始显示加入
        mid_pred = self.weight1 * out + self.weight2 * trans_bert_feats

        return tags, mid_pred, heads_outputs, types, gaz_match

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count,
                                gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bound, batch_type,
                                batch_bert,
                                bert_mask, trans_matrix):

        tags, slots, bound, types, _ = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count,
                                                     gaz_chars,
                                                     gaz_mask,
                                                     gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask,
                                                     trans_matrix)

        total_loss2 = self.crf.neg_log_likelihood_loss(slots, mask, batch_label)

        # type loss
        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        tmp1 = types.view(seq_len * batch_size, self.type_length)
        tmp2 = batch_type.view(seq_len * batch_size, -1).squeeze()
        type_loss = F.cross_entropy(tmp1, tmp2)

        total_loss2 += type_loss

        # bound loss
        tmp3 = bound.view(seq_len * batch_size, self.bound_length)
        tmp4 = batch_bound.view(seq_len * batch_size, -1).squeeze()
        bound_loss = F.cross_entropy(tmp3, tmp4)

        total_loss2 += bound_loss

        scores2, tag_seq2 = self.crf._viterbi_decode(slots, mask)

        return total_loss2, tag_seq2

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                gazchar_mask, mask, batch_bert, bert_mask, trans_matrix):

        tags, slots, bound, types, gaz_match = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz,
                                                             gaz_count,
                                                             gaz_chars, gaz_mask,
                                                             gazchar_mask, mask, word_seq_lengths, batch_bert,
                                                             bert_mask,
                                                             trans_matrix)

        scores, tag_seq = self.crf._viterbi_decode(slots, mask)

        return tag_seq, gaz_match


class I_S_Block(nn.Module):
    def __init__(self, hidden_size):
        super(I_S_Block, self).__init__()
        self.I_S_Attention = I_S_SelfAttention(hidden_size, 2 * hidden_size, hidden_size)
        self.I_Out = SelfOutput(hidden_size, 0.5)
        self.S_Out = SelfOutput(hidden_size, 0.5)
        self.I_S_Feed_forward = Intermediate_I_S(hidden_size, hidden_size)

    def forward(self, H_intent_input, H_slot_input, mask):
        H_slot, H_intent = self.I_S_Attention(H_intent_input, H_slot_input, mask)
        H_slot = self.S_Out(H_slot, H_slot_input)
        H_intent = self.I_Out(H_intent, H_intent_input)
        H_intent, H_slot = self.I_S_Feed_forward(H_intent, H_slot)

        return H_intent, H_slot


class I_S_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_slot = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_slot = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.value_slot = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(0.5)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, intent, slot, mask):
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(intent)
        mixed_key_layer = self.key(slot)
        mixed_value_layer = self.value(slot)

        mixed_query_layer_slot = self.query_slot(slot)
        mixed_key_layer_slot = self.key_slot(intent)
        mixed_value_layer_slot = self.value_slot(intent)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_slot = self.transpose_for_scores(mixed_query_layer_slot)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer_slot = self.transpose_for_scores(mixed_key_layer_slot)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        value_layer_slot = self.transpose_for_scores(mixed_value_layer_slot)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores_slot = torch.matmul(query_slot, key_slot.transpose(1,0))
        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2))
        attention_scores_slot = attention_scores_slot / math.sqrt(self.attention_head_size)
        attention_scores_intent = attention_scores + attention_mask

        attention_scores_slot = attention_scores_slot + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)

        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)

        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent, value_layer)

        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()
        context_layer_intent = context_layer_intent.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (self.out_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)
        return context_layer, context_layer_intent


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        super(Label_Attention, self).__init__()

        self.W_intent_emb = intent_emb.weight
        self.W_slot_emb = slot_emb.weight

    def forward(self, input_intent, input_slot):
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())
        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)

        return intent_res, slot_res


class Label_Attention2(nn.Module):
    def __init__(self, bound_emb, type_emb, slot_emb):
        super(Label_Attention2, self).__init__()

        self.W_bound_emb = bound_emb.weight
        self.W_intent_emb = type_emb.weight
        self.W_slot_emb = slot_emb.weight

    def forward(self, input_bound, input_intent, input_slot):
        bound_score = torch.matmul(input_bound, self.W_bound_emb.t())
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())
        bound_probs = nn.Softmax(dim=-1)(bound_score)
        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        bound_res = torch.matmul(bound_probs, self.W_bound_emb)
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)

        return bound_res, intent_res, slot_res


class Intermediate_I_S(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(Intermediate_I_S, self).__init__()
        self.dense_in = nn.Linear(hidden_size * 6, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm_I = LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm_S = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.5)

    def forward(self, hidden_states_I, hidden_states_S):
        hidden_states_in = torch.cat([hidden_states_I, hidden_states_S], dim=2)
        batch_size, max_length, hidden_size = hidden_states_in.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size)
        if torch.cuda.is_available():
            h_pad = h_pad.cuda()
        h_left = torch.cat([h_pad, hidden_states_in[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([hidden_states_in[:, 1:, :], h_pad], dim=1)
        hidden_states_in = torch.cat([hidden_states_in, h_left, h_right], dim=2)

        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states_I_NEW = self.LayerNorm_I(hidden_states + hidden_states_I)
        hidden_states_S_NEW = self.LayerNorm_S(hidden_states + hidden_states_S)
        return hidden_states_I_NEW, hidden_states_S_NEW


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BilinearLayer(nn.Module):
    def __init__(self, x_size, y_size, class_num):
        super(BilinearLayer, self).__init__()
        self.linear = nn.Linear(y_size, x_size * class_num)
        self.class_num = class_num

    def forward(self, x, y):
        Wy = self.linear(y)
        Wy = Wy.view(Wy.size(0), self.class_num, x.size(1))
        xWy = torch.sum(x.unsqueeze(1).expand_as(Wy) * Wy, dim=2)
        return xWy  # size = batch * class_num


class FusionLayer(nn.Module):
    def __init__(self, x_size, y_size, dropout_rate, class_num):
        super(FusionLayer, self).__init__()

        self.x_size = x_size
        self.y_size = y_size

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.linear_layer = nn.Linear(x_size, class_num)

        self._Wh = nn.Parameter(torch.Tensor(x_size, x_size))
        self._Wt = nn.Parameter(torch.Tensor(y_size, y_size))
        self._Wr = nn.Parameter(torch.Tensor(y_size, y_size))
        self._b = nn.Parameter(torch.Tensor(y_size))
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self._Wh)
        nn.init.kaiming_normal_(self._Wt)
        nn.init.kaiming_normal_(self._Wr)

        nn.init.normal_(self._b)

    def forward(self, x, y=None, dropout=True):
        left = torch.matmul(x, self._Wh)
        right = torch.matmul(y, self._Wt)
        ttt = left + right
        activated_outer_sum_bias = F.tanh(ttt)
        rate = torch.sigmoid(torch.matmul(activated_outer_sum_bias, self._Wr))

        fusion = rate * x + (1 - rate) * y
        if dropout:
            fusion = self.dropout_layer(fusion)
            # 效果不行
            # tmp1 = x.view(seq_len * batch_size, -1).squeeze()
            # tmp2 = y.view(seq_len * batch_size, -1).squeeze()
            # Wy = self.linear(tmp2)
            # Wy = Wy.view(Wy.size(0), 1, tmp1.size(1))
            # xWy = torch.sum(tmp1.unsqueeze(1).expand_as(Wy) * Wy, dim=2)
            # xx = xWy.view(batch_size, seq_len, 1)
            # rate = torch.sigmoid(xx)
            # merge_representation = torch.cat((x, y), dim=-1)
            # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
            # gated_converted_att_vis_embed = torch.mul(gate_value, y)
            # fusion = torch.cat((x, gated_converted_att_vis_embed), dim=-1)

        return self.linear_layer(fusion)
