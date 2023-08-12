# Add models (cancer only, cancer + organ task), DANN, Mixstyle

from torchvision import models
import torch
from torch import nn

network = models.efficientnet_b0(weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1')
in_features = network.classifier[-1].in_features
network.classifier[-1] = nn.Linear(in_features, 4)

# for single, multi organ model, only classify class of cancer
single_task_model = nn.DataParallel(network)

# for multi organ model, classify class of cancer and which organ
# for multi organ model, classify class of cancer and which organ
class cancer_classifier(nn.Module):
  def __init__(self):
    super(cancer_classifier, self).__init__()
    self.classifier = nn.Linear(1280,4)
    self.dropout = nn.Dropout(p = 0.2)

  def forward(self, x):
    x = self.dropout(x)
    x = self.classifier(x)
    return x

class organ_classifier(nn.Module):
  def __init__(self):
    super(organ_classifier, self).__init__()
    self.classifier = nn.Linear(1280,3)
    self.dropout = nn.Dropout(p = 0.2)

  def forward(self, x):
    x = self.dropout(x)
    x = self.classifier(x)
    return x

class multiTaskModel(nn.Module):
  def __init__(self):
    super(multiTaskModel, self).__init__()
    self.backbone = models.efficientnet_b0(weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1')
    self.backbone.classifier = nn.Identity()
    self.cancer_classifier = cancer_classifier()
    self.organ_classifier = organ_classifier()

  def forward(self, x):
    features = self.backbone(x)
    logit_cancer = self.cancer_classifier(features)
    logit_organ = self.organ_classifier(features)
    return logit_cancer, logit_organ

multi_network = multiTaskModel()
multi_task_model = nn.DataParallel(multi_network)

# Use DANN with gradient reversal layer

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1)

class organ_reversal_classifier(nn.Module):
  def __init__(self):
    super(organ_reversal_classifier, self).__init__()
    self.classifier = nn.Linear(1280,3)
    self.dropout = nn.Dropout(p = 0.2)

  def forward(self, x):
    x = GradReverse.apply(x)
    x = self.dropout(x)
    x = self.classifier(x)
    return x

class DANN_Model(nn.Module):
  def __init__(self):
    super(DANN_Model, self).__init__()
    self.backbone = models.efficientnet_b0(weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1')
    self.backbone.classifier = nn.Identity()
    self.cancer_classifier = cancer_classifier()
    self.organ_classifier = organ_reversal_classifier()

  def forward(self, x):
    features = self.backbone(x)
    logit_cancer = self.cancer_classifier(features)
    logit_organ = self.organ_classifier(features)
    return logit_cancer, logit_organ

dann_network = DANN_Model()
DANN_model = nn.DataParallel(dann_network)


# Use Mixstyle for gradient generalization

# class MixStyleModel(nn.Module):
  # or def function to modify efficientnet childrens