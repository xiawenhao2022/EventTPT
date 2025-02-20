import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.EventTPT.utils import combine_tokens, recover_tokens, token2feature, feature2token
from lib.models.EventTPT.vit import VisionTransformer
from lib.models.layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)


class AdaptiveFusionModule(nn.Module):
    def __init__(self, rgb_channels, event_channels, mid_channels, out_channels, reduction=16):
        super(AdaptiveFusionModule, self).__init__()
        self.rgb_channels = rgb_channels
        self.event_channels = event_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.sigmoid = nn.Sigmoid()

        self.channel_attention_fc = nn.Sequential(
            nn.Linear(768, 768 // reduction),  # 输入特征数改为768
            nn.ReLU(inplace=True),
            nn.Linear(768 // reduction, 768)  # 确保输出特征数回到768
        )

    def forward(self, rgb, event):
        batch_size, seq_len, channels = rgb.size()  # 假设event具有相同的尺寸

        x = torch.cat((rgb, event), dim=1)  # [batch_size, seq_len, channels]

        # 计算全局对比度，这里使用整个张量的标准差作为对比度的度量
        global_contrast = x.view(x.size(0), -1).std(dim=1)  # [batch_size, 1]
        global_contrast = torch.mean(global_contrast)  # 计算所有样本的对比度平均值

        # 将全局对比度转换为权重
        global_weight = 1 - self.sigmoid(global_contrast)

        x_rgb = global_weight * rgb

        event_flatten = event.view(batch_size * seq_len, channels)  # [batch_size * seq_len, channels]
        # 应用通道注意力网络
        channel_att = self.channel_attention_fc(event_flatten)  # [batch_size * seq_len, 768 // reduction]
        channel_att = channel_att.relu_()  # 应用ReLU激活函数
        # 重塑channel_att以匹配原始event的形状
        channel_att = channel_att.view(batch_size, seq_len, channels)  # [batch_size, seq_len, channels]

        # 应用通道注意力权重到event张量
        event_with_attention = event * channel_att



        # 将加权后的RGB和应用通道注意力的Event相加
        x = x_rgb + event_with_attention

        return x


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h * w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1

        return self.conv1x1(x0)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None, adaptive_fusion=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        if adaptive_fusion:
            self.adaptive_fusion_module = AdaptiveFusionModule(
                rgb_channels=64,  # 假定RGB图像有3个通道
                event_channels=64,  # 根据实际情况设置Event图像的通道数
                mid_channels=64,  # 中间层通道数
                out_channels=64,  # 输出通道数，这里假设与RGB通道数相同
            )
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type in ['vipt_shaw', 'eventtpt_deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'eventtpt_deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def adaptive_fusion(self, rgb_features, event_features):
        """
        自适应融合RGB和Event特征。

        参数:
        - rgb_features: RGB模态的特征图
        - event_features: Event模态的特征图

        返回:
        - fused_features: 融合后的特征图
        """
        # 计算RGB模态的对比度并应用自适应权重
        contrast = self.compute_contrast(rgb_features)
        rgb_weight = self.sigmoid(contrast)

        # 调整Event特征图的尺寸以匹配RGB特征图（如果需要）
        event_features = F.interpolate(event_features, size=rgb_features.size()[2:])

        # 融合RGB和Event特征
        fused_features = rgb_weight * rgb_features + (1 - rgb_weight) * event_features

        # 通过卷积层进一步处理融合后的特征
        fused_features = self.fusion_module.conv2(fused_features)

        return fused_features

    def compute_contrast(self, x):
        """
        计算特征图的对比度。

        参数:
        - x: 输入特征图

        返回:
        - contrast: 对比度图
        """
        mean = torch.mean(x, 1, keepdim=True)
        std = torch.std(x, unbiased=False)
        contrast = std / (mean + 1e-5)  # 添加小常数避免除零
        return contrast

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb

        z = self.patch_embed(z)
        x = self.patch_embed(x)

        z_dte = self.patch_embed_prompt(z_dte)
        x_dte = self.patch_embed_prompt(x_dte)

        '''input prompt: by adding to rgb tokens'''
        if self.prompt_type in ['vipt_shaw', 'eventtpt_deep']:
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))
            z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
            x_dte_feat = token2feature(self.prompt_norms[0](x_dte))
            z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
            x_feat = torch.cat([x_feat, x_dte_feat], dim=1)
            z_feat = self.prompt_blocks[0](z_feat)
            x_feat = self.prompt_blocks[0](x_feat)
            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            z_prompted, x_prompted = z_dte, x_dte

            z = self.adaptive_fusion_module(z, z_dte)
            x = self.adaptive_fusion_module(x, x_dte)
        else:
            z = z + z_dte
            x = x + x_dte

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                if self.prompt_type in ['eventtpt_deep']:
                    x_ori = x
                    # recover x to go through prompt blocks
                    lens_z_new = global_index_t.shape[1]
                    lens_x_new = global_index_s.shape[1]
                    z = x[:, :lens_z_new]
                    x = x[:, lens_z_new:]
                    if removed_indexes_s and removed_indexes_s[0] is not None:
                        removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                        pruned_lens_x = lens_x - lens_x_new
                        pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                        x = torch.cat([x, pad_x], dim=1)
                        index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                        C = x.shape[-1]
                        x = torch.zeros_like(x).scatter_(dim=1,
                                                         index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                                         src=x)
                    x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                    x = torch.cat([z, x], dim=1)

                    # prompt
                    x = self.prompt_norms[i - 1](x)  # todo
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:, :]
                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)

                    z_prompted = self.prompt_norms[i](z_prompted)
                    x_prompted = self.prompt_norms[i](x_prompted)
                    z_prompt_feat = token2feature(z_prompted)
                    x_prompt_feat = token2feature(x_prompted)

                    z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
                    x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)
                    z_feat = self.prompt_blocks[i](z_feat)
                    x_feat = self.prompt_blocks[i](x_feat)

                    z = feature2token(z_feat)
                    x = feature2token(x_feat)
                    z_prompted, x_prompted = z, x

                    x = combine_tokens(z, x, mode=self.cat_mode)
                    # re-conduct CE
                    x = x_ori + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64),
                                             src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, )

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=True)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
