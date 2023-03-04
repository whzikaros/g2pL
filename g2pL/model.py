"""
implement of LEBERT
本代码抽取于transfortmers3.4.0
zj为transfortmers3.4.0基础上另外transfortmers3.4.0加进去的代码。
模型结构可参考：https://blog.csdn.net/myboyliu2007/article/details/115611660

"""
import torch
from torch import nn
from typing import Optional, Tuple

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput, load_tf_weights_in_bert, BertModel
from transformers.file_utils import ModelOutput


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        """
        无改动。
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertLayer(nn.Module): 
    """
    修改了该模块，将word embedding信息加进去。
    
    该模块是transformer单层的encoder部分，
    主要包括三个模块(结合transformer的encoder图看):
    BertAttention: 多头自注意力机制和其残差，norm
    BertIntermediate: 前馈神经网络
    BertOutput: 残差，norm
    参考博客：https://blog.csdn.net/myboyliu2007/article/details/115611660
    """
    def __init__(self, config, has_word_attn=False, seq_len=None): #has_word_attn参数zj
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config) #transformer核心的自注意力机制
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention #3.4.0新加，设置文件中没有则为false
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
            
        ## zj,here we add a attention for matched word
        self.has_word_attn = has_word_attn
        if self.has_word_attn:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.act = nn.Tanh()

            self.word_transform = nn.Linear(config.word_embed_dim, config.hidden_size) 
            self.word_word_weight = nn.Linear(config.hidden_size, config.hidden_size) 
            attn_W = torch.zeros(config.hidden_size, config.hidden_size) # dc × dc
            self.attn_W = nn.Parameter(attn_W) 
            self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)
            self.fuse_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
            #对所有输出的隐藏层使用全连接层转换成一个隐藏层
            if seq_len:
                # 使用1D卷积
                self.conv1=nn.Conv1d(in_channels=seq_len,out_channels=1,kernel_size=1)
                self.relu=nn.ReLU(inplace=True)
                
        self.intermediate = BertIntermediate(config) #
        self.output = BertOutput(config) #

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        input_word_embeddings=None,#zj
        input_word_mask=None,#zj
        bs=None, #batch_size
        poly_ids=None,  
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        """
        N: batch_size
        L: seq length 
        W: word size 5
        D: word_embedding dim 200 对应论文中的d2，表示每个词的embedding维度，200
        config.hidden_size: 对应论文中的dc，768
        
        Args:
            input_word_embedding: [N, L, W, D]
            input_word_mask: [N, L, W]
        """
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None: #此处无is_decoder，表示false
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(  
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        ) #layer_output shape : torch.Size([N, L, dc])

        if self.has_word_attn:
            assert input_word_mask is not None
                   
            layer_output_poly_hidden = layer_output[torch.arange(bs), poly_ids] # 得到[N, dc]
            layer_output_poly_hidden = layer_output_poly_hidden.unsqueeze(1) # [N, dc] -> [N, 1, dc]
            
            poly_input_word_embeddings = input_word_embeddings[torch.arange(bs), poly_ids] # 得到[N, W, D]
            poly_input_word_embeddings = poly_input_word_embeddings.unsqueeze(1) # [N, W, D] -> [N, 1, W, D]

            poly_input_word_mask = input_word_mask[torch.arange(bs), poly_ids] # 得到[N, W]
            poly_input_word_mask = poly_input_word_mask.unsqueeze(1) # [N, W] -> [N, 1, W]

            word_outputs = self.word_transform(poly_input_word_embeddings)  # [N, 1, W, D] × [dc, dc]-> [N, 1, W, dc]
            word_outputs = self.act(word_outputs) #公式2中的tanh
            word_outputs = self.word_word_weight(word_outputs) # [N, 1, W, dc] × [dc, dc] -> [N, 1, W, dc]
            word_outputs = self.dropout(word_outputs)

            #使用一维卷积
            hidden=self.conv1(layer_output.clone()) # [N, L, dc] -> [N, 1, dc]
            hidden=self.relu(hidden)

            alpha = torch.matmul(hidden.unsqueeze(2), self.attn_W)  # [N, 1, 1, dc] × [dc, dc]-> [N, 1, 1, dc]
            alpha = torch.matmul(alpha, torch.transpose(word_outputs, 2, 3))  # [N, 1, 1, dc] × [N, 1, dc, W] -> [N, 1, 1, W]

            alpha = alpha.squeeze(2)  # [N, 1, W]

            alpha = alpha + (1 - poly_input_word_mask.float()) * (-10000.0) 

            alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, 1, W]
            alpha = alpha.unsqueeze(-1)  # [N, 1, W, 1]

            weighted_word_embedding = torch.sum(word_outputs * alpha, dim=2)  # [N, 1, W, dc] 和 [N, 1, W, 1] -> [N, 1, dc] 
            layer_output_poly_hidden = layer_output_poly_hidden + weighted_word_embedding
            layer_output_poly_hidden = self.dropout(layer_output_poly_hidden)
            layer_output_poly_hidden = self.fuse_layernorm(layer_output_poly_hidden) 
            layer_output_poly_hidden = layer_output_poly_hidden.squeeze()
            
            #替换原来的隐藏层
            layer_output[torch.arange(bs), poly_ids] = layer_output_poly_hidden
            
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config, seq_len):
        super().__init__()
        self.config = config
        #zj
        self.add_layers = config.add_layers
                
        #将以下代码替换self.layer
        total_layers = []
        for i in range(config.num_hidden_layers):
            if i in self.add_layers:
                total_layers.append(BertLayer(config, True, seq_len))
            else:
                total_layers.append(BertLayer(config, False, seq_len))
        self.layer = nn.ModuleList(total_layers)
        #self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        input_word_embeddings=None, #zj
        input_word_mask=None, #zj
        batch_size=None,
        poly_ids=None,        
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,#zj
                    input_word_mask,#zj                    
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    input_word_embeddings,#zj
                    input_word_mask,#zj
                    batch_size,
                    poly_ids,                     
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # return BaseModelOutput(
        #     last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        # )
        #替换注释代码
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

class BertPooler(nn.Module):
    """
    无改动。
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    无改动。
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class WCBertModel(BertPreTrainedModel):
    def __init__(self, config, seq_len, add_pooling_layer=True):
        super(WCBertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, seq_len)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        matched_word_embeddings=None,
        matched_word_mask=None,
        batch_size=None,
        poly_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

        batch_size: N
        seq_length: L
        dim: D
        word_num: W
        boundary_num: B


        Args:
            input_ids: [N, L]
            attention_mask: [N, L]
            matched_word_embeddings: [B, L, W, D]
            matched_word_mask: [B, L, W]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            input_word_embeddings=matched_word_embeddings, #见BertLayer
            input_word_mask=matched_word_mask,
            batch_size=batch_size,
            poly_ids=poly_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class G2pL(BertPreTrainedModel):
    def __init__(self, config, pretrained_embeddings, num_labels, seq_len):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]  #词的数量
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = WCBertModel(config, seq_len)
        self.dropout = nn.Dropout(config.HP_dropout)
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

        ## init the embedding
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        #print("Load pretrained embedding from file.........")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            poly_ids=None
    ):
        #print(222222222,input_ids)
        #print(matched_word_ids)
        matched_word_embeddings = self.word_embeddings(matched_word_ids)

        #print(3333333333,matched_word_embeddings)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=matched_word_mask,
            batch_size=input_ids.size(0),
            poly_ids=poly_ids
        )

        hidden = outputs[0]
        batch_size = input_ids.size(0)
        poly_hidden = hidden[torch.arange(batch_size), poly_ids] #获取每条数据的poly_ids的位置的值
        # hidden = self.dropout(hidden)
        logits = self.classifier(poly_hidden)
        
        return logits