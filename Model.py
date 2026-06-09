import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic_module import BasicBlock


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet18Backbone(nn.Module):
    """ResNet-18 stem used by the original GAP implementation.

    The release code keeps this local implementation so the project can run in
    environments without extra vision libraries. It accepts optional ImageNet
    weights saved as a plain state_dict with standard ResNet-18 keys.
    """

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self._init_weights()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [ResNetBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResNetBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def load_pretrained_state_dict(self, path):
        if not path or not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            print(f"Skipped unexpected pretrained keys: {unexpected}")
        if missing:
            print(f"Missing pretrained keys: {missing}")
        return True


def _gate(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.Dropout2d(0.5),
        nn.BatchNorm2d(channels),
        nn.Sigmoid(),
    )


class My_model(nn.Module):
    """GAP model with the original public-facing class name.

    Default forward output remains compatible with the research scripts:
    ``bvp, hr, spo2, rr = model(x)``.
    Set ``return_details=True`` to get features and task representations needed
    by the GAP losses.
    """

    def __init__(self, people_num=1, pretrained_path="./pre_encoder/resnet18-5c106cde.pth"):
        super().__init__()
        resnet = ResNet18Backbone()
        resnet.load_pretrained_state_dict(pretrained_path)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.gate_hr = _gate(256)
        self.gate_bvp = _gate(256)
        self.gate_rr = _gate(256)
        self.gate_spo = _gate(256)
        self.gate_person = _gate(256)

        self.part_hr = copy.deepcopy(resnet.layer4)
        self.part_bvp = copy.deepcopy(resnet.layer4)
        self.part_rr = copy.deepcopy(resnet.layer4)
        self.part_spo = copy.deepcopy(resnet.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hr = nn.Linear(512, 1)
        self.spo = nn.Linear(512, 1)
        self.rf = nn.Linear(512, 1)
        self.person_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, people_num),
        )

        self.up1_bvp = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2_bvp = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 64, [1, 1], downsample=1),
        )
        self.up3_bvp = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [2, 1], downsample=1),
        )
        self.up4_bvp = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 1, [1, 1], downsample=1, islast=True),
        )

    def count_parameters(self, grad):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == grad)

    def calculate_training_parameter_ratio(self):
        trainable = self.count_parameters(True)
        frozen = self.count_parameters(False)
        ratio = trainable / max(frozen + trainable, 1)
        print("Non-trainable parameters:", frozen)
        print("Trainable parameters (M):", trainable / (1024 ** 2))
        print("Ratio:", ratio)
        return ratio

    @staticmethod
    def _pool_repr(x):
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    @staticmethod
    def _fuse_with_person(task_feature, person_feature):
        fused = task_feature + task_feature * person_feature
        return F.layer_norm(fused, fused.shape[1:])

    def forward(self, input, mode="mssdg", return_details=False):
        batch_size = input.size(0)

        x = self.relu(self.bn1(self.conv1(input)))
        x = self.layer1(x)
        feat1 = x
        x = self.layer2(x)
        feat2 = x
        x = self.layer3(x)
        feat3 = x

        person_feature = 2 * self.gate_person(x) * x
        hr_feature = 2 * self.gate_hr(x) * x
        bvp_feature = 2 * self.gate_bvp(x) * x
        rr_feature = 2 * self.gate_rr(x) * x
        spo_feature = 2 * self.gate_spo(x) * x

        if mode == "ttpa":
            hr_feature = self._fuse_with_person(hr_feature, person_feature)
            bvp_feature = self._fuse_with_person(bvp_feature, person_feature)
            rr_feature = self._fuse_with_person(rr_feature, person_feature)
            spo_feature = self._fuse_with_person(spo_feature, person_feature)

        em_hr = self.part_hr(hr_feature)
        em_bvp = self.part_bvp(bvp_feature)
        em_rr = self.part_rr(rr_feature)
        em_spo = self.part_spo(spo_feature)

        hr = self.hr(self.avgpool(em_hr).reshape(batch_size, -1))
        rr = self.rf(self.avgpool(em_rr).reshape(batch_size, -1))
        spo = self.spo(self.avgpool(em_spo).reshape(batch_size, -1))

        bvp = self.up4_bvp(self.up3_bvp(self.up2_bvp(self.up1_bvp(em_bvp)))).squeeze(1)
        person_repr = self._pool_repr(person_feature)
        person_logits = self.person_head(person_repr)

        if not return_details:
            return bvp, hr, spo, rr

        return {
            "preds": {
                "bvp": bvp,
                "hr": hr,
                "spo2": spo,
                "rr": rr,
            },
            "block_feats": [feat1, feat2, feat3],
            "task_reprs": {
                "hr": self._pool_repr(hr_feature),
                "bvp": self._pool_repr(bvp_feature),
                "spo2": self._pool_repr(spo_feature),
                "rr": self._pool_repr(rr_feature),
            },
            "person_repr": person_repr,
            "person_logits": person_logits,
        }


def freeze_model_BN(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.training = False
    return model


def freeze_model(model, freeze=True):
    if freeze:
        for name, param in model.named_parameters():
            train_head = any(key in name for key in ["bvp", "hr", "spo", "rr", "rf"])
            param.requires_grad = train_head and ("part" not in name and "gate" not in name)
        for name, param in model.named_parameters():
            if "bn" in name:
                param.requires_grad = True
    else:
        for _, param in model.named_parameters():
            param.requires_grad = True
    return model
