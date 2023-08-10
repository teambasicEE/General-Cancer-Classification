# Add models (cancer only, cancer + organ), DANN, Mixstyle
# 주의할 점 : Multi-task model 어떻게 만들어야 할지? 데이터의 경우 label에 class, organ에 organ 오도록 설정해주었음

from torchvision import models
from torch import nn

network = models.efficientnet_b0(weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1')
in_features = network.classifier[-1].in_features
network.classifier[-1] = nn.Linear(in_features, 4)

single_organ_model = nn.DataParallel(network)
