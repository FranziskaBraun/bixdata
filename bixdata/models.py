import torch
from torch import nn
from .fusion_models import GatedModel, ConcatModel, SingleModel, GatedMultimodalUnit
import torch.nn.functional as F
from sktdementia.base_data import get_norm_value_tensor
from torch_geometric.nn import GCNConv
# from .constants import *
# torch.manual_seed(SEED)


def pool_padded(input, mask):
    # mask = mask.unsqueeze(-1).expand(input.size()).float()
    # mask = torch.logical_not(mask).float()
    mask = mask[:, 0, :]
    mask = mask.unsqueeze(-1).expand(input.size())
    sum_embeddings = torch.sum(input * mask, 1)
    sum_mask = mask.sum(1)
    return sum_embeddings / sum_mask


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[], dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=None):
        super(MLP, self).__init__()
        layers = nn.ModuleList()

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization_fn:
                layers.append(normalization_fn(hidden_dim))
            layers.append(activation_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))
        # layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class AttentionCLSHead(nn.Module):
    def __init__(self, att_dim=768, att_heads=1):
        super().__init__()

        self.seq_query = torch.nn.init.xavier_normal_(torch.nn.Parameter(torch.zeros(1, att_dim)))
        self.attention = nn.MultiheadAttention(embed_dim=att_dim, num_heads=att_heads, batch_first=True, dropout=0.5)

    def forward(self, x, attn_mask):
        """x is a sequence of wav2vec 2.0 embeddings, batch dim is dim 0 """
        # pool, _ = self.attention(self.seq_query, x, x)
        pool, _ = self.attention(torch.stack([self.seq_query for x in range(x.shape[0])]), x, x)
        pool = pool.squeeze(1)  # remove seq dim
        # pool = self.layer_norm(pool)
        return pool


class W2v2CLSAttentionPooling(nn.Module):
    def __init__(self, emb_dim=512, att_heads=1):
        super().__init__()
        self.gelu = nn.GELU()
        self.attention = nn.MultiheadAttention(emb_dim, att_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, attn_mask):
        # pool w.r.t. the mean vector of the sequence
        x = torch.nan_to_num(x, nan=0.0)
        key = pool_padded(x, attn_mask)
        # pool, _ = self.attention(torch.mean(x, dim=1, keepdim=True), x, x)
        pool, _ = self.attention(torch.unsqueeze(key, 1), x, x, key_padding_mask=attn_mask)
        pool = self.layer_norm(pool.squeeze(dim=1))
        pool = self.gelu(pool)
        return pool


class MultiheadAttentionPooling(nn.Module):
    def __init__(self, emb_dim=768, att_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, att_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, q, k, v, attn_mask=None, need_weights=False):
        att_output, att_weights = self.attention(q, k, v, attn_mask=attn_mask, need_weights=need_weights)
        att_weights = att_weights * attn_mask if need_weights and attn_mask else att_weights
        pool = pool_padded(att_output, attn_mask) if attn_mask is not None else att_output.mean(dim=1)
        pool = self.layer_norm(pool)
        return pool, att_weights


class CombinedConcatModule(nn.Module):
    def __init__(self, emb_dim=768, att_heads=1, out_dim=2, dropout=0.1):
        super().__init__()
        self.attention_mod = MultiheadAttentionPooling(emb_dim, att_heads)
        self.concat_mod = ConcatModel(ft1_size=emb_dim, ft2_size=emb_dim, output_dim=out_dim, mlp_dropout=dropout)

    def forward(self, t, a, att_mask_t, att_mask_a):
        # pool w.r.t. the mean vector of the sequence
        t_attention_output, _ = self.attention_mod(t, att_mask_t)
        a_attention_output, _ = self.attention_mod(a, att_mask_a)
        x = self.concat_mod(t_attention_output, a_attention_output)
        return x


class CombinedGMUModule(nn.Module):
    def __init__(self, emb_dim=768, att_heads=1, out_dim=2, dropout=0.1):
        super().__init__()
        self.attention_mod_text = AttentionBlock(emb_dim, att_heads)
        self.attention_mod_audio = AttentionBlock(emb_dim, att_heads)
        self.gated_mod = GatedModel(ft1_size=emb_dim, ft2_size=emb_dim, output_dim=out_dim, mlp_dropout=dropout)

    def forward(self, t, a):
        text_attention_output = self.attention_mod_text(t, t, t)
        audio_attention_output = self.attention_mod_audio(a, a, a)
        x = self.gated_mod(text_attention_output, audio_attention_output)
        return x


class SingleModule(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=512, att_heads=1, out_dim=2, dropout=0.1):
        super().__init__()
        # self.projection_layer = nn.Linear(emb_dim, emb_dim)
        self.attention_mod = AttentionBlock(emb_dim, att_heads)
        self.single_mod = SingleModel(ft1_size=emb_dim, hidden_dim=hidden_dim, output_dim=out_dim, mlp_dropout=dropout)
        # self.mlp_mod = MLP(in_dim=emb_dim, out_dim=out_dim, hidden_dims=[512, 512])

    def forward(self, x):
        # x = self.projection_layer(x)
        # x = self.relu(x)
        x = self.attention_mod(x, x, x)
        x = self.single_mod(x)
        return x


class ScalarOneHotModel(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=512, test_dim=61, out_dim=1):
        super(ScalarOneHotModel, self).__init__()
        self.test_dim = test_dim
        # Linear layers for the scalar and feature vector
        # self.scalar_fc = nn.Linear(scalar_dim, hidden_dim)  # For the scalar input
        self.scalar_fc = nn.Linear(test_dim, hidden_dim)  # For the scalar input
        self.feature_fc = nn.Linear(feature_dim, hidden_dim)  # For the feature vector
        self.attention = AttentionBlock(feature_dim)

        # Fully connected layers after fusion
        # self.fusion_fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.fusion_fc = nn.Linear(hidden_dim * 2, test_dim)
        # self.test_fc = nn.Linear(hidden_dim, test_dim)
        self.output_fc = nn.Linear(test_dim, out_dim)  # Output a single scalar value

    def forward(self, s, f):
        # Pass the scalar through its own linear layer
        s = F.one_hot(s.long(), num_classes=self.test_dim).float()
        s = self.scalar_fc(s)
        # s = F.relu(self.scalar_fc(s))  # ReLU after linear

        f = self.attention(f, f, f)
        f = self.feature_fc(f)
        # f = F.relu(self.feature_fc(f))  # ReLU after linear

        # Fuse the transformed scalar and feature vector by concatenating them
        x = torch.cat((s, f), dim=1)
        x = torch.relu(x)

        # Further process the fused input through additional layers
        x = self.fusion_fc(x)
        x = torch.relu(x)
        # x = self.test_fc(x)
        # x = torch.relu(x)
        x = self.output_fc(x)

        return x


class ScalarFusionModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(ScalarFusionModel, self).__init__()
        reduction1 = hidden_dims[0]
        reduction2 = hidden_dims[1]
        self.scalar_fc = nn.Linear(1, reduction1)
        self.attention = AttentionBlock(f_dim)
        self.feature_fc = nn.Linear(f_dim, reduction1)
        self.fusion_fc = nn.Linear(reduction1 * 2, reduction2)
        self.output_fc = nn.Linear(reduction2, o_dim)

    def forward(self, s, f):
        # Pass the scalar input
        s = self.scalar_fc(s.view(-1, 1))
        s = torch.relu(s)
        # Pass the feature vector
        f = self.attention(f, f, f)
        f = torch.relu(f)
        f = self.feature_fc(f)
        f = torch.relu(f)
        # Concatenate
        x = torch.cat((s, f), dim=1)
        # Process fused representation
        x = self.fusion_fc(x)
        x = torch.relu(x)
        # Final output layer
        x = self.output_fc(x)
        return x


class EmbeddingModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(EmbeddingModel, self).__init__()
        reduction1 = hidden_dims[0]
        reduction2 = hidden_dims[1]
        self.attention = AttentionBlock(f_dim)
        self.feature_fc = nn.Linear(f_dim, reduction1)
        self.fusion_fc = nn.Linear(reduction1, reduction2)
        self.output_fc = nn.Linear(reduction2, o_dim)

    def forward(self, s, f):
        # Pass the feature vector
        f = self.attention(f, f, f)
        f = torch.relu(f)
        f = self.feature_fc(f)
        f = torch.relu(f)
        # Process fused representation
        x = self.fusion_fc(f)
        x = torch.relu(x)
        # Final output layer
        x = self.output_fc(x)
        return x


class LargeScalarFusionModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(LargeScalarFusionModel, self).__init__()
        reduction1 = hidden_dims[0]
        reduction2 = hidden_dims[1]
        self.scalar_fc = nn.Linear(1, reduction1)
        self.attention = AttentionBlock(f_dim)
        self.feature_fc1 = nn.Linear(f_dim, 768)
        self.feature_fc2 = nn.Linear(768, reduction1)
        self.fusion_fc = nn.Linear(reduction1 * 2, reduction2)
        self.output_fc = nn.Linear(reduction2, o_dim)

    def forward(self, s, f):
        # Pass the scalar input
        s = self.scalar_fc(s.view(-1, 1))
        s = torch.relu(s)
        # Pass the feature vector
        f = self.attention(f, f, f)
        f = torch.relu(f)
        f = self.feature_fc1(f)
        f = torch.relu(f)
        f = self.feature_fc2(f)
        f = torch.relu(f)
        # Concatenate
        x = torch.cat((s, f), dim=1)
        # Process fused representation
        x = self.fusion_fc(x)
        x = torch.relu(x)
        # Final output layer
        x = self.output_fc(x)
        return x


class CrossAttFusionModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(CrossAttFusionModel, self).__init__()
        reduction1 = hidden_dims[0]
        reduction2 = hidden_dims[1]
        self.scalar_fc = nn.Linear(1, reduction1)
        self.attention = AttentionBlock(f_dim)
        self.feature_fc = nn.Linear(f_dim, reduction1)
        self.fusion_fc = nn.Linear(reduction1 * 2, reduction2)
        self.output_fc = nn.Linear(reduction2, o_dim)

    def forward(self, s, f1, f2):
        # Pass the scalar input
        s = self.scalar_fc(s.view(-1, 1))
        s = torch.relu(s)
        # Pass the feature vector
        f = self.attention(f1, f2, f2)
        f = torch.relu(f)
        f = self.feature_fc(f)
        f = torch.relu(f)
        x = torch.cat((s, f), dim=1)
        # Process fused representation
        x = self.fusion_fc(x)
        x = torch.relu(x)
        # Final output layer
        x = self.output_fc(x)
        return x


class EarlyFusionModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(EarlyFusionModel, self).__init__()
        reduction1 = hidden_dims[0]
        reduction2 = hidden_dims[1]
        self.scalar_fc = nn.Linear(1, reduction1)
        self.attention1 = AttentionBlock(f_dim)
        self.attention2 = AttentionBlock(f_dim)
        self.feature_fc1 = GatedMultimodalUnit(f_dim, f_dim, f_dim)
        self.feature_fc2 = nn.Linear(f_dim, reduction1)
        self.fusion_fc = nn.Linear(reduction1 * 2, reduction2)
        self.output_fc = nn.Linear(reduction2, o_dim)

    def forward(self, s, f1, f2):
        # Pass the scalar input
        s = self.scalar_fc(s.view(-1, 1))
        s = torch.relu(s)
        # Pass the feature vector
        f1 = self.attention1(f1, f1, f1)
        f2 = self.attention2(f2, f2, f2)
        # f = torch.cat((f1, f2), dim=1)
        f = self.feature_fc1(f1, f2)
        f = torch.relu(f)
        f = self.feature_fc2(f)
        f = torch.relu(f)
        x = torch.cat((s, f), dim=1)
        # Process fused representation
        x = self.fusion_fc(x)
        x = torch.relu(x)
        # Final output layer
        x = self.output_fc(x)
        return x


class CrossEmbeddingModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(CrossEmbeddingModel, self).__init__()
        reduction1 = hidden_dims[0]
        reduction2 = hidden_dims[1]
        self.attention = AttentionBlock(f_dim)
        self.feature_fc = nn.Linear(f_dim, reduction1)
        self.fusion_fc = nn.Linear(reduction1, reduction2)
        self.output_fc = nn.Linear(reduction2, o_dim)

    def forward(self, s, f1, f2):
        # Pass the feature vector
        f = self.attention(f1, f2, f2)
        f = torch.relu(f)
        f = self.feature_fc(f)
        f = torch.relu(f)
        # Process fused representation
        x = self.fusion_fc(f)
        x = torch.relu(x)
        # Final output layer
        x = self.output_fc(x)
        return x


class LateFusionModel(nn.Module):
    def __init__(self, f_dim=768, o_dim=1, hidden_dims=[256, 64]):
        super(LateFusionModel, self).__init__()
        self.scalar_model = ScalarFusionModel(f_dim=f_dim, o_dim=o_dim, hidden_dims=hidden_dims)

    def forward(self, s, f1, f2):
        # Pass the scalar input
        f = torch.cat((f1, f2), dim=1)
        x = self.scalar_model(s, f)
        return x


class ExpertFusionModel(nn.Module):
    def __init__(self, f_dim=768, hidden_dims=[256, 64], o_dim=1, num_experts=2):
        super(ExpertFusionModel, self).__init__()
        self.num_experts = num_experts
        self.attention1 = AttentionBlock(f_dim)
        self.attention2 = AttentionBlock(f_dim)
        # Define experts (ScalarFusionModels)
        self.experts = nn.ModuleList(
            [ScalarFusionModel(f_dim=f_dim, hidden_dims=hidden_dims, o_dim=o_dim) for _ in range(num_experts)])
        # Gating mechanism
        self.gate_fc = nn.Linear(f_dim*2, num_experts)
        # Output layer to produce final output
        self.output_fc = nn.Linear(o_dim, o_dim)

    def forward(self, s, f1, f2):
        # Input features (f1, f2) processing
        f1 = self.attention1(f1, f1, f1)
        f2 = self.attention2(f2, f2, f2)
        # Combine the features into a single tensor
        features = torch.cat((f1, f2), dim=-1)  # Concatenate along feature dimension
        # Compute gating weights (softmax to get probability distribution over experts)
        gate_scores = self.gate_fc(features)
        gate_weights = F.softmax(gate_scores, dim=1)  # Shape: (batch_size, num_experts)
        # Expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(s, f1)  # Assuming ScalarFusionModel takes `s` and one feature (e.g., f1)
            expert_outputs.append(expert_output)
        # Stack the expert outputs and combine them using the gate weights
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: (batch_size, num_experts, o_dim)
        weighted_expert_output = torch.sum(gate_weights.unsqueeze(2) * expert_outputs,
                                           dim=1)  # Shape: (batch_size, o_dim)
        # Final output
        x = torch.relu(weighted_expert_output)
        x = self.output_fc(x)
        return x


class FinetuneModule(nn.Module):
    def __init__(self, pretrained_model, emb_dim=768, att_heads=1, out_dim=2, dropout=0.1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.single_mod = SingleModule(emb_dim=emb_dim, att_heads=att_heads, out_dim=out_dim, dropout=dropout)

    def forward(self, x):
        mask = x["attention_mask"].float()
        mask = [m * torch.transpose(m.unsqueeze(dim=0), 0, 1) for m in mask]
        mask = torch.stack(mask)
        x = self.pretrained_model(**x)[0]
        x = self.single_mod(x, mask)
        return x


class CrossAttentionModule(nn.Module):
    def __init__(self, emb_dim=768, att_heads=1, out_dim=2, dropout=0.1):
        super().__init__()
        self.attention_mod_text = AttentionBlock(emb_dim, att_heads)
        self.attention_mod_audio = AttentionBlock(emb_dim, att_heads)
        # self.single_mod = SingleModel(ft1_size=emb_dim, output_dim=out_dim, mlp_dropout=dropout)
        self.concat_mod = ConcatModel(ft1_size=emb_dim, ft2_size=emb_dim, output_dim=out_dim, mlp_dropout=dropout)

    def forward(self, t, a):

        t_att_out = self.attention_mod_text(query=t, key=a, value=a)
        a_att_out = self.attention_mod_audio(query=a, key=t, value=t)
        x = self.concat_mod(t_att_out, a_att_out)
        return x


class ConcatModule(nn.Module):
    def __init__(self, emb_dim=768, out_dim=2, dropout=0.1):
        super().__init__()
        self.concat_mod = ConcatModel(ft1_size=emb_dim, ft2_size=emb_dim, output_dim=out_dim, mlp_dropout=dropout)

    def forward(self, t, a):
        x = self.concat_mod(t.mean(dim=1), a.mean(dim=1))
        return x


class OnewayCrossAttentionModule(nn.Module):
    def __init__(self, emb_dim=768, att_heads=2, out_dim=2, dropout=0.1):
        super().__init__()
        self.attention_mod_text = AttentionBlock(emb_dim, att_heads)
        self.single_mod = SingleModel(ft1_size=emb_dim, output_dim=out_dim, mlp_dropout=dropout)

    def forward(self, t, a):
        x = self.attention_mod_text(query=t, key=a, value=a)
        x = self.single_mod(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim=768, att_heads=1):
        super().__init__()
        self.MHA = nn.MultiheadAttention(emb_dim, att_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(emb_dim)
        # self.feed_forward = nn.Linear(emb_dim, emb_dim)
        # self.layernorm2 = nn.LayerNorm(emb_dim)

    def forward(self, query, key, value):
        x_MHA, _ = self.MHA(query=query, key=key, value=value, need_weights=False)
        # x = x_MHA + query # residual
        x = self.layernorm1(x_MHA.mean(dim=1))
        # x, _ = torch.max(x_MHA, dim=1) # max pooling
        # x = self.layernorm1(pool_padded(x_MHA, mask)) # padded pooling
        return x


class NormScalarFusionModel(nn.Module):
    def __init__(self, subtest_models, o_dim, norm, tests):
        super(NormScalarFusionModel, self).__init__()
        self.norm = norm
        self.tests = tests
        self.input_encoders = nn.ModuleList(subtest_models)
        self.test_layer = nn.Linear(len(subtest_models), 9)
        self.output_layer = nn.Linear(9, o_dim)

    def forward(self, age, iq, scalars, features):
        raw = []
        for i, feature in enumerate(features):
            raw.append(self.input_encoders[i](scalars[i], feature))
        norm = []
        for i, value in enumerate(raw):
            norm.append(get_norm_value_tensor(self.norm, self.tests[i], value.squeeze(), iq, age))
        norm = torch.cat(norm, dim=1)
        norm = torch.relu(norm)
        total = self.test_layer(norm)
        total = torch.relu(total)
        total = self.output_layer(total)
        return total


class ConcatenationModel(nn.Module):
    def __init__(self, feature_dim):
        super(ConcatenationModel, self).__init__()
        self.attention = AttentionBlock(feature_dim)
        self.fc1 = nn.Linear(feature_dim + 1, 256)  # Feature vector + scalar
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, scalar, feature_vector):
        scalar = scalar.view(-1, 1)
        feature_vector = self.attention(feature_vector, feature_vector, feature_vector)
        x = torch.cat([scalar, feature_vector], dim=1)  # Concatenate along feature dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ParallelModel(nn.Module):
    def __init__(self, feature_dim):
        super(ParallelModel, self).__init__()
        # Scalar Branch
        self.scalar_branch = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.attention = AttentionBlock(feature_dim)

        # Feature Branch
        self.feature_branch = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, scalar, feature_vector):
        scalar = scalar.view(-1, 1)
        scalar_out = self.scalar_branch(scalar)
        feature_vector = self.attention(feature_vector, feature_vector, feature_vector)
        feature_out = self.feature_branch(feature_vector)
        x = torch.cat([scalar_out, feature_out], dim=1)
        x = self.fusion(x)
        return x


class AttentionFusionModel(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusionModel, self).__init__()
        self.scalar_fc = nn.Linear(1, 256)
        self.attention = AttentionBlock(feature_dim)
        self.feature_fc = nn.Linear(feature_dim, 256)
        self.attention_fc = nn.Linear(256, 1)
        self.final_fc = nn.Sequential(
            nn.Linear(feature_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, scalar, feature_vector):
        scalar = scalar.view(-1, 1)
        scalar_embed = F.relu(self.scalar_fc(scalar))  # Scalar embedding
        feature_vector = self.attention(feature_vector, feature_vector, feature_vector)
        feature_embed = F.relu(self.feature_fc(feature_vector))  # Feature embedding

        # Attention scores
        attention_scores = F.softmax(self.attention_fc(scalar_embed + feature_embed), dim=1)

        # Weighted fusion
        weighted_scalar = scalar * attention_scores
        weighted_features = feature_vector * attention_scores

        # Concatenate and process
        x = torch.cat([weighted_scalar, weighted_features], dim=1)
        x = self.final_fc(x)
        return x


class FiLMModel(nn.Module):
    def __init__(self, feature_dim):
        super(FiLMModel, self).__init__()
        # Scalar to modulation parameters
        self.gamma_fc = nn.Linear(1, 1280)
        self.beta_fc = nn.Linear(1, 1280)
        self.attention = AttentionBlock(feature_dim)
        # Feature processing
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, scalar, feature_vector):
        scalar = scalar.view(-1, 1)
        gamma = self.gamma_fc(scalar)  # Scaling factor
        beta = self.beta_fc(scalar)  # Shifting factor
        feature_vector = self.attention(feature_vector, feature_vector, feature_vector)
        # Apply modulation
        modulated_features = gamma * feature_vector + beta

        # Process modulated features
        x = self.fc(modulated_features)
        return x


class GNNFusionModel(nn.Module):
    def __init__(self, feature_dim):
        super(GNNFusionModel, self).__init__()
        # GCN Layers
        self.attention = AttentionBlock(feature_dim)
        self.gcn1 = GCNConv(feature_dim + 1, 512)  # Scalar + Feature Vector
        self.gcn2 = GCNConv(512, 256)

        # Final Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output regression value
        )

    def forward(self, scalar, feature_vector, edge_index):
        scalar = scalar.view(-1, 1)
        # Combine scalar and feature_vector into node features
        feature_vector = self.attention(feature_vector, feature_vector, feature_vector)
        node_features = torch.cat([scalar, feature_vector], dim=1)
        x = self.gcn1(node_features, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)

        # Final output
        x = self.fc(x)
        return x