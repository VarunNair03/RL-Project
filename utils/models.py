import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True):
        super(FeatureExtractor, self).__init__()

        # Supported architectures
        arch_dict = { 
            # ResNets
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            # EfficientNets
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
            "efficientnet_b5": models.efficientnet_b5,
            "efficientnet_b6": models.efficientnet_b6,
            "efficientnet_b7": models.efficientnet_b7,
            # MobileNets
            # "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
            # ConvNeXts
            "convnext_tiny": models.convnext_tiny,
            "convnext_small": models.convnext_small,
            "convnext_base": models.convnext_base,
            "convnext_large": models.convnext_large,
            # Vision Transformers
            "vit_base_16": models.vit_b_16,
            "vit_base_32": models.vit_b_32,
            "vit_large_16": models.vit_l_16,
            "vit_large_32": models.vit_l_32,
            # "vit_huge_14": models.vit_h_14,
            # Swin Transformer V2
            "swinv2_tiny": models.swin_v2_t,
            "swinv2_small": models.swin_v2_s,
            "swinv2_base": models.swin_v2_b,
        }

        assert arch in arch_dict, (
            f"Unsupported architecture: {arch}. Choose from {list(arch_dict.keys())}"
        )

        # Load the model
        self.model = arch_dict[arch](weights="DEFAULT" if pretrained else None)

        # Modify the model to remove classifier and get feature dimension
        if "resnet" in arch:
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif "efficientnet" in arch:
            self.feature_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
        elif "mobilenet" in arch:
            self.feature_dim = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()
        elif "convnext" in arch:
            self.feature_dim = self.model.classifier[2].in_features  # [LayerNorm, Flatten, Linear]
            self.model.classifier = self.model.classifier[:2]
        elif "vit" in arch:
            self.feature_dim = self.model.heads.head.in_features
            self.model.heads.head = nn.Identity()
        elif "swin" in arch:
            self.feature_dim = self.model.head.in_features
            self.model.head = nn.Identity()

    def get_feature_dim(self):
        return self.feature_dim
    
    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)  # Flatten output


class DQN(nn.Module):
    def __init__(self, h, w, outputs, feature_dim):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_dim + 81, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=outputs)
        )

    def forward(self, x):
        return self.classifier(x)


def model_factory(arch, outputs, pretrained=True):
    """
    Factory method to create the feature extractor and DQN model.

    Args:
        arch (str): Model architecture name.
        outputs (int): Number of output actions for DQN.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        feature_extractor (nn.Module): Selected feature extractor model.
        dqn (nn.Module): Corresponding DQN model.
    """
    feature_extractor = FeatureExtractor(arch, pretrained)
    dqn = DQN(feature_extractor.feature_dim, outputs)
    return feature_extractor, dqn
