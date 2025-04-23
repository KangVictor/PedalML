import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import sys

class FxNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.batchNorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.batchNorm2 = nn.BatchNorm2d(12)
        self.fc1 = nn.Linear(12*29*18, 120)
        self.batchNorm3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 60)
        self.batchNorm4 = nn.BatchNorm1d(60)
        self.out = nn.Linear(60, self.n_classes)

    def forward(self, t):
        t = self.conv1(t)
        t = self.batchNorm1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2, 2)
        t = self.conv2(t)
        t = self.batchNorm2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, 2, 2)
        t = t.view(-1, 12*29*18)
        t = self.fc1(t)
        t = self.batchNorm3(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = self.batchNorm4(t)
        t = F.relu(t)
        t = self.out(t)
        return t

# # Step 2: Dynamically create the module hierarchy: model.models.FxNet
# model_module = types.ModuleType("model")
# models_submodule = types.ModuleType("model.models")
# setattr(models_submodule, "FxNet", FxNet)
# setattr(model_module, "models", models_submodule)

# # Step 3: Inject into sys.modules
# sys.modules["model"] = model_module
# sys.modules["model.models"] = models_submodule

# # Step 4: Load model
# model_path = "/fx_classifier_models/20201211_fxnet_poly_disc_noTS9_best"
# # model_path = "/fx_classifier_models/20201027_fxnet_poly_disc_best"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fxnet_model = torch.load(model_path, map_location=device, weights_only=False)
# fxnet_model.eval()
