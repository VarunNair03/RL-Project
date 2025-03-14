import torch.nn as nn
import torchvision


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        effnet = torchvision.models.efficientnet_b0(torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        effnet.eval() 
        # self.features = effnet.features  # Remove classifier
        self.features = nn.Sequential(*list(effnet.children())[:-1])  # Remove classifier
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten output

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1280 + 81, out_features=1024), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=outputs)
        )
    
    def forward(self, x):
        return self.classifier(x)
    

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         resnet50 = torchvision.models.resnet50(pretrained=True)
#         resnet50.eval()  # Ensure dropout is not used
#         self.features = nn.Sequential(*list(resnet50.children())[:-1])  # Remove final FC layer
    
#     def forward(self, x):
#         x = self.features(x)
#         return x.view(x.size(0), -1)  # Flatten output

# class DQN(nn.Module):
#     def __init__(self, outputs):
#         super(DQN, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=2048 + 81, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=outputs)
#         )
    
#     def forward(self, x):
#         return self.classifier(x)

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         densenet = torchvision.models.densenet121(pretrained=True)
#         densenet.eval()
#         self.features = densenet.features  # Extract features before classification layer
    
#     def forward(self, x):
#         x = self.features(x)
#         return x.view(x.size(0), -1)  # Flatten output

# class DQN(nn.Module):
#     def __init__(self, outputs):
#         super(DQN, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=1024 + 81, out_features=1024),  # DenseNet-121 outputs 1024 features
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=outputs)
#         )

#     def forward(self, x):
#         return self.classifier(x)


# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         convnext = torchvision.models.convnext_tiny(pretrained=True)
#         convnext.eval()
#         self.features = nn.Sequential(*list(convnext.children())[:-1])  # Remove classifier
    
#     def forward(self, x):
#         x = self.features(x)
#         return x.view(x.size(0), -1)  # Flatten output

# class DQN(nn.Module):
#     def __init__(self, outputs):
#         super(DQN, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=768 + 81, out_features=1024),  # ConvNeXt-Tiny outputs 768 features
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(in_features=1024, out_features=outputs)
#         )

#     def forward(self, x):
#         return self.classifier(x)
