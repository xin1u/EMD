import torch
import torch.nn as nn
from torchvision import models
import os


class Vgg19Features(nn.Module):
    """VGG-19 feature extractor for contrastive loss.
    Uses layers {relu1_1, relu2_1, relu3_1, relu4_1, relu5_1}."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=False)
        vgg_path = os.path.join(os.path.dirname(__file__), 'vgg19-dcbb9e9d.pth')
        if os.path.exists(vgg_path):
            vgg.load_state_dict(torch.load(vgg_path, map_location='cpu'))
        else:
            vgg = models.vgg19(pretrained=True)
        self.features = vgg.features
        self.feature_layers = [2, 7, 12, 21, 30]  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if (i + 1) in self.feature_layers:
                feats.append(x)
            if (i + 1) == self.feature_layers[-1]:
                break
        return feats


class ContrastRegularizationLoss(nn.Module):
    """Contrast regularization loss (Eq.8 in paper).

    Pushes restored image features closer to GT features,
    and away from degraded input features in VGG-19 feature space.
    """
    def __init__(self, tau=0.07, weights=None):
        super().__init__()
        self.vgg = Vgg19Features()
        self.vgg.eval()
        self.tau = tau
        # weight for each VGG layer, follow [7]
        if weights is None:
            self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            self.weights = weights

    def forward(self, restored, gt, degraded):
        """
        Args:
            restored: network output f(x)
            gt: ground truth y
            degraded: blurry input x
        """
        feat_restored = self.vgg(restored)
        feat_gt = self.vgg(gt)
        feat_degraded = self.vgg(degraded)

        loss = 0.
        for l in range(len(feat_restored)):
            # L1 distance
            d_pos = torch.mean(torch.abs(feat_restored[l] - feat_gt[l]))
            d_neg = torch.mean(torch.abs(feat_restored[l] - feat_degraded[l]))

            # contrastive formulation from Eq. 8
            numerator = torch.exp(-d_pos / self.tau)
            denominator = numerator + torch.exp(-d_neg / self.tau)
            loss += -self.weights[l] * torch.log(numerator / denominator + 1e-8)

        return loss


if __name__ == '__main__':
    loss_fn = ContrastRegularizationLoss()
    x = torch.randn(1, 3, 64, 64)
    y = torch.randn(1, 3, 64, 64)
    z = torch.randn(1, 3, 64, 64)
    print('contrast loss:', loss_fn(x, y, z).item())
