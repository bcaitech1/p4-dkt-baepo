from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
from transformers.models.bert.modeling_bert import (BertConfig, BertEncoder,
                                                    BertModel)

from dkt.dataloader import FeatureInfo, FeatureType


class DktBert(nn.Module):
    def __init__(self,
        device: torch.device,
        n_test: int,
        n_questions: int,
        n_tag: int,
        hidden_dim: int,
        n_layers: int,
        n_heads,
        max_seq_len: int,
    ):
        super(DktBert, self).__init__()
        self.device = device

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            max_position_embeddings=max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)
       
        self.activation = nn.Sigmoid()


    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)    # [64, 20] = [batch size, max_seq_len]

        # 신나는 embedding
        
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat([embed_interaction,
        
                           embed_test,
                           embed_question,
        
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):

    def __init__(self,
        device: torch.device,
        n_test: int,
        n_questions: int,
        n_tag: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        drop_out: float
    ):
        super(LSTMATTN, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.drop_out = drop_out

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class ContinousReshape(nn.Module):
    def __init__(self, embedding_dim: int):
        super(ContinousReshape, self).__init__()
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(2)
        output = self.linear(x)
        return output


class DktNewBert(nn.Module):
    def __init__(self,
            feature_info: List[FeatureInfo],
            max_seq_len: int,
            embedding_dim: int, # Embedding Dim
            n_heads: int,       # Transformer head
            n_layers: int,      # Transformer layer
            hidden_dim: int,    # FC Layer Dim
        ):
        super(DktNewBert, self).__init__()
        embedding_list = []
        self.hidden_dim = hidden_dim
        
        for target_feature in feature_info:
            if target_feature.feature_type == FeatureType.Categorical:
                embedding_list.append(nn.Sequential(
                    nn.Embedding(target_feature.num_class + 1, embedding_dim),  # 왜 +1이었더라...
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim)
                ))
            else:       # Continous
                embedding_list.append(nn.Sequential(
                    ContinousReshape(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim)
                ))
        self.embeddings = nn.modules.ModuleList(embedding_list)

        self.comb_proj = nn.Linear(embedding_dim * (len(self.embeddings)), hidden_dim)

        self.config = BertConfig( 
            3, # not used
            hidden_size=hidden_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            max_position_embeddings=max_seq_len          
        )
        self.encoder = BertModel(self.config)
        self.fc = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

        # 제일 의심가는 부분은 max_seq_len으로 자르는 부분.
        # 현재 Task는 뒷부분을 추론하는 것인데, 앞뒤 구분없이 여기저기 잘라서 학습시키니
        # 뒤쪽을 잘 못 맞추는 것
    
    def forward(self, inputs: List[torch.Tensor]):
        features = inputs[:-1]
        mask = inputs[-1]
        batch_size = inputs[0].size(0)

        feature_embedding = []
        for feature, embed_layer in zip(features, self.embeddings):
            feature_embedding.append(
                embed_layer(feature)
            )

        feature_inputs = torch.cat(feature_embedding, 2)
        feature_inputs = self.comb_proj(feature_inputs)

        # Bert
        encoded = self.encoder(inputs_embeds=feature_inputs, attention_mask=mask)
        out: torch.Tensor = encoded[0]
        # out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)
        
        return preds
            

        
