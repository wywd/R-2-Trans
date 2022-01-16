import torch
import torch.nn as nn

import numpy as np
# from scipy.spatial.distance import pdist, squareform

from models.modeling_mask import VisionTransformer
# from utils.ib import calculate_MI


def get_mask_nums(att_maps, λ=1.0):
    batch_mean = torch.mean(att_maps, dim=1, keepdim=True) * λ
    p_mean = torch.mean((att_maps < batch_mean).type(torch.float).sum(dim=1)) / att_maps.shape[-1]
    p_nums = int(p_mean * att_maps.shape[-1])
    return p_nums


class R2Trans(nn.Module):  # for embedding layer
    def __init__(self, config, args, num_classes, mask_scope=0.5):
        super(R2Trans, self).__init__()
        self.device = config.device
        self.img_size = args.img_size
        self.num_classes = num_classes
        self.base_model_one = VisionTransformer(config, self.img_size, num_classes, vis=True, zero_head=True)
        # if args.pretrained_dir is not None:
        #     print('===> load pretrained base model : {}'.format(args.pretrained_dir))
        #     self.base_model_one.load_from(np.load(args.pretrained_dir))

        self.num_patches = int(self.img_size / config.patches['size'][0]) ** 2
        self.h = int(np.sqrt(self.num_patches))
        self.k_scope = int(self.num_patches * mask_scope)   # for ablation experiment
        self.norm = nn.LayerNorm(self.num_patches)

        self.hidden_concat = config.hidden_size * 2
        self.classifier = nn.Linear(self.hidden_concat, num_classes)  # extra auxiliary branch, only for train

    def forward(self, x, labels=None):
        if labels is not None:
            # BDMM part
            loss1, logits1, cls_feature1, att_maps = self.base_model_one(x, labels)
            att_maps = torch.stack(att_maps, dim=0).detach()  # 12, B, 12, 785, 785

            # mean+mean
            att_maps = att_maps.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
            att_maps = self.norm(att_maps.sum(dim=1))  # B, 784
            cls_maps = torch.sigmoid(att_maps)  # B, 784

            num_mask = get_mask_nums(cls_maps, λ=1.0)
            # print('===> ', num_mask)

            sorted_index = torch.argsort(cls_maps, dim=1, descending=False)  # B  low to high
            max_index = sorted_index[:, num_mask:] + 1

            loss2, logits2, cls_feature2, _ = self.base_model_one(x, labels, mask=max_index)
            cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
            logits_concat = self.classifier(cls_concat)
            loss_concat = nn.CrossEntropyLoss()(logits_concat.view(-1, self.num_classes), labels.view(-1))

            #  MI part

            # return loss1, loss2, loss_concat, loss_ixz, logits1, logits2, logits_concat
            return loss1, loss2, loss_concat, logits1, logits2, logits_concat

        else:
            logits1, cls_feature1, weight_map = self.base_model_one(x)  # stage1 (paper use this for calc acc)
            att_maps = torch.stack(weight_map, dim=0).detach()  # 12, B, 12, 785, 785

            att_maps = att_maps.mean(dim=2).mean(dim=0)[:, 1:, 1:]  # B, 785, 785 -> B, 784, 784
            att_maps = self.norm(att_maps.sum(dim=1))  # B, 784
            cls_maps = torch.sigmoid(att_maps)  # B, 784

            num_mask = get_mask_nums(cls_maps, λ=1.0)
            # print('===> ', num_mask)

            sorted_index = torch.argsort(cls_maps, dim=1, descending=False)
            # max_index = sorted_index[:, self.k_scope:] + 1
            max_index = sorted_index[:, num_mask:] + 1

            logits2, cls_feature2, _ = self.base_model_one(x, mask=max_index)  # stage2
            cls_concat = torch.cat((cls_feature1, cls_feature2), dim=1)
            logits_concat = self.classifier(cls_concat)
            return logits1, logits2, logits_concat, weight_map  # weight_map from stage1

    def get_params(self):
        freshlayer_params = list(self.classifier.parameters()) + list(self.base_model_one.head.parameters())
        ftlayer_params_ids = list(map(id, freshlayer_params))
        ftlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())
        return ftlayer_params, freshlayer_params


