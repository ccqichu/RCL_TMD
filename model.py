from typing import Optional

from torch import Tensor
from transformers import CLIPModel
# from transformers import BertConfig
# from transformers.models.bert.modeling_bert import BertLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


# from backbone import build_backbone

# from transformers import BertTokenizer, BertModel


# ============================================================================
# Helper functions for CID-DIMM
# ============================================================================

def masked_mean(x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
    """
    Compute masked mean over sequence dimension.
    Args:
        x: [B, L, D] tensor
        pad_mask: [B, L] boolean tensor where True indicates padding positions
    Returns:
        [B, D] tensor
    """
    if pad_mask is None:
        return x.mean(dim=1)
    keep = (~pad_mask).float().unsqueeze(-1)  # [B, L, 1]
    denom = keep.sum(dim=1).clamp_min(1.0)
    return (x * keep).sum(dim=1) / denom


def make_neg_index(batch_size: int, device):
    """
    Generate negative sample indices for contrastive learning.
    Ensures idx[i] != i for all i.
    Args:
        batch_size: batch size
        device: torch device
    Returns:
        [B] tensor of shuffled indices
    """
    idx = torch.randperm(batch_size, device=device)
    # Avoid idx[i]==i, use circular shift if collision exists
    if torch.any(idx == torch.arange(batch_size, device=device)):
        idx = torch.roll(idx, shifts=1, dims=0)
    return idx


class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:

            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values


# ============================================================================
# CID Module: Consistent/Inconsistent Decomposition with soft mask
# ============================================================================

class CIDModule(nn.Module):
    """
    CID Module with soft mask, temperature annealing, and consistency losses.

    Inputs:
        - T_tok: [B, Lt, 512] CLIP text last_hidden_state
        - V_tok: [B, Lv, 768] CLIP vision last_hidden_state
        - attn_mask: [B, Lt] attention mask (1=valid, 0=pad)

    Outputs:
        - T: [B, Lt, 768] projected text features
        - V_con: [B, Lv, 768] consistent vision patches
        - V_inc: [B, Lv, 768] inconsistent vision patches
        - m_v: [B, Lv] consistency mask
        - loss_ratio: scalar, ratio loss
        - loss_itm: scalar, image-text matching mask loss
    """

    def __init__(self, text_dim=512, vision_dim=768, hidden_dim=768,
                 rho=0.3, delta=0.1, tau0=1.0, tau_min=0.4, decay=0.9995):
        super(CIDModule, self).__init__()

        # Project text from 512 to 768
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Hyperparameters
        self.rho = rho  # target ratio for consistent patches
        self.delta = delta  # margin for ITM loss
        self.tau0 = tau0  # initial temperature
        self.tau_min = tau_min  # minimum temperature
        self.decay = decay  # temperature decay rate

        # Temperature annealing: maintain global step as buffer
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

    def forward(self, T_tok, V_tok, attn_mask):
        """
        Args:
            T_tok: [B, Lt, 512] text features from CLIP
            V_tok: [B, Lv, 768] vision features from CLIP
            attn_mask: [B, Lt] attention mask (1=valid, 0=pad)

        Returns:
            T, V_con, V_inc, m_v, loss_ratio, loss_itm
        """
        B, Lt, _ = T_tok.shape
        Lv = V_tok.size(1)
        device = T_tok.device

        # 1. Project text to 768 dimensions
        T = self.text_proj(T_tok)  # [B, Lt, 768]
        V = V_tok  # [B, Lv, 768]

        # 2. Create padding mask (True = padding)
        pad_mask = (attn_mask == 0)  # [B, Lt]

        # 3. Temperature annealing
        if self.training:
            self.global_step += 1
        tau = max(self.tau_min, self.tau0 * (self.decay ** self.global_step.item()))

        # 4. Compute alignment scores
        # A = T @ V^T / tau -> [B, Lt, Lv]
        A = torch.matmul(T, V.transpose(1, 2)) / tau  # [B, Lt, Lv]

        # s_v = max over text tokens -> [B, Lv]
        s_v = A.max(dim=1).values  # [B, Lv]

        # 5. Soft mask using softmax
        m_v = F.softmax(s_v, dim=-1)  # [B, Lv]

        # 6. Split into consistent and inconsistent parts
        V_con = m_v.unsqueeze(-1) * V  # [B, Lv, 768]
        V_inc = (1 - m_v.unsqueeze(-1)) * V  # [B, Lv, 768]

        # 7. L_ratio: encourage mask mean to be close to rho
        loss_ratio = (m_v.mean() - self.rho) ** 2

        # 8. L_itm: image-text matching mask loss
        # Create negative samples by shuffling batch
        idx_neg = make_neg_index(B, device)
        V_neg = V[idx_neg]  # [B, Lv, 768]

        # Compute negative alignment
        A_neg = torch.matmul(T, V_neg.transpose(1, 2)) / tau  # [B, Lt, Lv]
        s_v_neg = A_neg.max(dim=1).values  # [B, Lv]
        m_v_neg = F.softmax(s_v_neg, dim=-1)  # [B, Lv]

        # Loss: encourage m_v_neg < m_v by at least delta
        loss_itm = F.relu(m_v_neg.mean() - m_v.mean() + self.delta)

        return T, V_con, V_inc, m_v, loss_ratio, loss_itm


# ============================================================================
# DIMM Module: Disentangled Interaction and Multimodal Modeling
# ============================================================================

class DIMMModule(nn.Module):
    """
    DIMM Module for reasoning over consistent/inconsistent interactions.

    Inputs:
        - T: [B, Lt, 768] projected text features
        - V_con: [B, Lv, 768] consistent vision patches
        - V_inc: [B, Lv, 768] inconsistent vision patches
        - pad_mask: [B, Lt] padding mask (True=pad)

    Outputs:
        - z_cid: [B, 768] unified CID representation
    """

    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1, num_encoder_layers=1):
        super(DIMMModule, self).__init__()

        # Multi-head attention for consistent interaction
        self.mha_con = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Multi-head attention for inter-modality reasoning (inconsistent)
        self.mha_inter = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Transformer encoder for intra-modality reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Final projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, T, V_con, V_inc, pad_mask):
        """
        Args:
            T: [B, Lt, 768] text features
            V_con: [B, Lv, 768] consistent vision patches
            V_inc: [B, Lv, 768] inconsistent vision patches
            pad_mask: [B, Lt] padding mask (True=pad)

        Returns:
            z_cid: [B, 768] unified representation
        """
        # Ensure pad_mask matches T's sequence length
        # This handles cases where sequence lengths vary across batches
        if pad_mask.size(1) != T.size(1):
            # Adjust pad_mask to match T's actual sequence length
            if pad_mask.size(1) > T.size(1):
                # Truncate pad_mask
                pad_mask = pad_mask[:, :T.size(1)]
            else:
                # Extend pad_mask with True (padding) values
                batch_size = pad_mask.size(0)
                extra_len = T.size(1) - pad_mask.size(1)
                extra_padding = torch.ones(batch_size, extra_len,
                                          dtype=pad_mask.dtype,
                                          device=pad_mask.device).bool()
                pad_mask = torch.cat([pad_mask, extra_padding], dim=1)

        # 1. Consistent interaction: text attends to consistent patches
        T_add, _ = self.mha_con(
            query=T,
            key=V_con,
            value=V_con,
            key_padding_mask=None  # Vision has no padding
        )
        T_enh = self.ln1(T + T_add)  # [B, Lt, 768]

        # Aggregate to vector using masked mean
        z_con = masked_mean(T_enh, pad_mask)  # [B, 768]

        # 2. Inter-modality reasoning: enhanced text attends to inconsistent patches
        T2, _ = self.mha_inter(
            query=T_enh,
            key=V_inc,
            value=V_inc,
            key_padding_mask=None
        )
        z_inter = masked_mean(T2, pad_mask)  # [B, 768]

        # 3. Intra-modality reasoning: self-attention on enhanced text
        T_ctx = self.transformer_encoder(
            T_enh,
            src_key_padding_mask=pad_mask
        )  # [B, Lt, 768]

        # Ensure pad_mask still matches T_ctx (in case transformer changes it)
        if T_ctx.size(1) != pad_mask.size(1):
            pad_mask = pad_mask[:, :T_ctx.size(1)]

        z_intra = masked_mean(T_ctx, pad_mask)  # [B, 768]

        # 4. Combine all three pathways
        z_cid = self.mlp(torch.cat([z_con, z_inter, z_intra], dim=-1))  # [B, 768]

        return z_cid


class RCLMuFN(nn.Module):
    def __init__(self, args):
        super(RCLMuFN, self).__init__()
        self.model = CLIPModel.from_pretrained("/home/user/chengtaiyu/models/clip-vit-base-patch32")
        # BERT-related initialization (disabled).
        # self.config = BertConfig.from_pretrained("/home/user/chengtaiyu/models/bert-base-uncased")
        # self.config.hidden_size = 768
        # self.config.num_attention_heads = 8
        # self.trans = MultimodalEncoder(self.config, layer_number=args.layers)
        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.image_size)
            self.image_linear =  nn.Linear(args.image_size, args.image_size)
        else:
            self.text_linear =  nn.Sequential(
                nn.Linear(args.text_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear =  nn.Sequential(
                nn.Linear(args.image_size, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
        self.classifier_fuse = nn.Linear(args.image_size , args.label_number)
        self.cross_att = CrossAttention(feature_dim=768, dropout_prob=0.1)
        self.loss_fct = nn.CrossEntropyLoss()
        # BERT/ResNet branches (disabled).
        # self.tokenizer = BertTokenizer.from_pretrained("/home/user/chengtaiyu/models/bert-base-uncased")
        # self.bert_model = BertModel.from_pretrained("/home/user/chengtaiyu/models/bert-base-uncased")
        # self.backbone = build_backbone(args)
        self.d_model = 768
        self.nheads = 8
        self.dim_feedforward = 2048
        # self.txt = nn.Sequential(nn.Linear(self.d_model, self.d_model),
        #                          nn.ReLU(),
        #                          nn.LayerNorm(self.d_model)
        #                          )
        self.txt2 = nn.Sequential(nn.Linear(self.d_model*2, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.LayerNorm(self.d_model)
                                 )
        self.vis2 = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                 nn.ReLU(),
                                  nn.Linear(self.d_model, self.d_model),
                                 nn.LayerNorm(self.d_model)
                                 )
        # self.text_self = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        # self.text_cross = TransformerCrossLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        # self.vis_self = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        # self.vis_cross = TransformerCrossLayer(self.d_model, self.nheads, dim_feedforward=self.dim_feedforward)
        # self.imtxt_cross = TransformerCrossLayer(768, self.nheads, dim_feedforward=self.dim_feedforward)
        self.attetion_block = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                            nn.ReLU(),
                                            nn.Linear(self.d_model, self.d_model),
                                            nn.Sigmoid())
        self.mlp_layer = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                       nn.ReLU(),
                                       nn.Linear(self.d_model, self.d_model),
                                       nn.LayerNorm(self.d_model))
        # self.input_proj = nn.Conv2d(self.dim_feedforward, 768, kernel_size=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ========================================================================
        # CID-DIMM Integration
        # ========================================================================
        # Instantiate CID and DIMM modules
        self.cid = CIDModule(
            text_dim=512,
            vision_dim=768,
            hidden_dim=768,
            rho=0.3,
            delta=0.1,
            tau0=1.0,
            tau_min=0.4,
            decay=0.9995
        )
        self.dimm = DIMMModule(
            hidden_dim=768,
            num_heads=8,
            dropout=0.1,
            num_encoder_layers=1
        )

        # Alignment and connection layers
        self.cid_proj = nn.Linear(768, 768, bias=True)
        self.cid_ln = nn.LayerNorm(768)
        self.res_ln = nn.LayerNorm(768)

        # Alpha parameter for gradual integration (initialized to 0 for reversibility)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Loss weights
        self.lambda_ratio = 1.0
        self.lambda_itm = 0.5

    def forward(self, inputs, batch, labels):
        output = self.model(**inputs,output_attentions=True)

        # 提取对应的特征
        text_features = output['text_model_output']['last_hidden_state']  # 128，77，512
        image_features = output['vision_model_output']['last_hidden_state']  # 128，50，768
        text_feature = output['text_model_output']['pooler_output'] # 64，512
        image_feature = output['vision_model_output']['pooler_output'] # 64，768
        text_feature = self.text_linear(text_feature)  # 64，768
        image_feature = self.image_linear(image_feature)  # 64,768

        text_list, image_list, label_list, id_list, samples = batch

        # ========================================================================
        # CID-DIMM Pipeline: Insert after extracting last_hidden_state
        # ========================================================================
        # Run CID module
        T, V_con, V_inc, m_v, loss_ratio, loss_itm = self.cid(
            text_features,
            image_features,
            inputs["attention_mask"]
        )

        # Create padding mask (True = pad) with correct sequence length matching T
        # Handle variable sequence lengths robustly
        actual_seq_len = T.size(1)  # Actual sequence length from CID output
        attn_mask = inputs["attention_mask"]

        if attn_mask.size(1) >= actual_seq_len:
            # Truncate attention mask to match T's length
            pad_mask = (attn_mask[:, :actual_seq_len] == 0)  # [B, actual_seq_len]
        else:
            # Extend attention mask if T is longer (rare case)
            batch_size = attn_mask.size(0)
            # Use existing mask + assume rest is padding
            pad_mask_base = (attn_mask == 0)
            extra_len = actual_seq_len - attn_mask.size(1)
            extra_padding = torch.ones(batch_size, extra_len,
                                      dtype=torch.bool,
                                      device=attn_mask.device)
            pad_mask = torch.cat([pad_mask_base, extra_padding], dim=1)

        # Run DIMM module
        z_cid = self.dimm(T, V_con, V_inc, pad_mask)  # [B, 768]

        # Project and normalize CID output
        cid_hat = self.cid_ln(self.cid_proj(z_cid))  # [B, 768]
        # ========================================================================

        # ResNet/BERT feature extraction (disabled).
        # features, pos = self.backbone(samples.to(inputs['input_ids'].device))
        # src, mask = features[-1].decompose()  # 32,2048,7,7
        #
        # # resnet50
        # src = self.input_proj(src)  # 64,768,7,7
        # pooled_features = self.pool(src)  # [batch_size, 768, 1, 1]
        # res_features = pooled_features.view(pooled_features.size(0), -1)  # [32, 768]
        # # bert
        # encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
        # encoded_input = encoded_input.to(inputs['input_ids'].device)
        # with torch.no_grad():
        #     outputs_bert = self.bert_model(**encoded_input)
        #     last_hidden_states = outputs_bert.last_hidden_state  # 64,56,768
        #     pooler_outputs = outputs_bert.pooler_output  # 64,768
        # bert_text_features = self.txt(pooler_outputs)  # 64,768

        # SFIM (disabled): keep original code commented for reference.
        # image_t = self.imtxt_cross(tgt=res_features, memory=bert_text_features)  # 32,768
        # text_im = self.imtxt_cross(tgt=bert_text_features, memory=res_features)  # 32,768
        # Use CLIP features directly.
        image_t = image_feature
        text_im = text_feature

        # RCLM (simplified): use CLIP features directly.
        # text_feature2 = self.txt2(torch.cat([text_feature, text_im], dim=-1))  # 32,768
        # text_feature2 = text_feature2.unsqueeze(1)  # 32,1,768
        # txt_cat = self.text_self(torch.stack([text_feature, text_im], dim=1))  # 32,2,768
        # txt_output = self.text_cross(tgt=text_feature2, memory=txt_cat)  # 32,1,768
        # txt_output = txt_output.squeeze(1)  # 32,768
        #
        # image_feature2 = self.vis2(torch.cat([image_feature, image_t],dim=-1))  # 32,768
        # image_feature2 = image_feature2.unsqueeze(1)  # 32,1,768
        # image_cat = self.vis_self(torch.stack([image_feature, image_t], dim=1))  # 32,2,768
        # image_output = self.vis_cross(tgt=image_feature2, memory=image_cat)  # 32,1,768
        # image_output = image_output.squeeze(1)  # 32,768
        txt_output = text_feature
        image_output = image_feature

        txt_out = self.cross_att(txt_output, image_output, image_output)  # 32,768
        image_out = self.cross_att(image_output, txt_output, txt_output)  # 32,768
        res_bert = 0.6 * image_out + 0.4 * txt_out  # 32,768
        # res_bert = image_t + text_im

        # ========================================================================
        # CID-DIMM Connection: Align and integrate with residual connection
        # ========================================================================
        # Apply alignment with learnable alpha (initialized to 0 for reversibility)
        res_new = self.res_ln(res_bert + self.alpha * cid_hat)  # [B, 768]
        # ========================================================================

        # CLIP-View Feature Fusion
        cross_feature_text = self.cross_att(text_feature, image_feature, image_feature)  # 32,768
        cross_feature_image = self.cross_att(image_feature, text_feature, text_feature)  # 32,768
        fuse_feature = 0.7 * cross_feature_text + 0.3 * cross_feature_image

        # MuFFM: Now using res_new (integrated with CID-DIMM)
        att = self.attetion_block(torch.cat([fuse_feature, res_new], dim=-1))  # 32,768
        output = 0.5 * fuse_feature + 0.5 * (att * self.mlp_layer(torch.cat([fuse_feature, res_new], dim=-1)))  # 32,768

        # Predict
        logits_fuse = self.classifier_fuse(output)  # 64,2  output
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)  # 64,2

        score = fuse_score

        outputs = (score,) # (64,2)
        if labels is not None:
            loss_fuse = self.loss_fct(logits_fuse, labels)
            # ====================================================================
            # CID-DIMM Loss Integration
            # ====================================================================
            # Combine classification loss with CID consistency losses
            loss = loss_fuse + self.lambda_ratio * loss_ratio + self.lambda_itm * loss_itm
            # ====================================================================
            outputs = (loss,) + outputs
        return outputs

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerCrossLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_attn=False,kdim=None,vdim=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=kdim,vdim=vdim,batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_attn = return_attn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
