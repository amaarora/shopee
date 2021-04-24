import timm
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale=30.0,
        margin=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class Shopee_Model(nn.Module):
    def __init__(
        self, arch, num_class, scale=30, margin=0.5, easy_margin=False, **model_kwargs
    ):
        super().__init__()
        self.backbone = timm.create_model(arch, num_classes=num_class, **model_kwargs)
        in_ftrs = self.backbone.get_classifier().in_features
        self.arc_margin = ArcMarginProduct(
            in_ftrs,
            num_class,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
            ls_eps=0.0,
        )

    def forward(self, image, label, extract_feat=False):
        if extract_feat:
            return self.backbone(image)
        out = self.backbone(image)
        out = self.arc_margin(out, label)
        return out