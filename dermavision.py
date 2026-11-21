"""DermaVision AI - shared code: class metadata, model, preprocessing, Grad-CAM.

Imported by train.py (training / evaluation) and app.py (Streamlit demo).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# HAM10000 diagnosis codes. Alphabetical order = label index used everywhere.
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_INFO = {
    "akiec": {"name": "Actinic keratosis / intraepithelial carcinoma", "risk": "Pre-malignant",
              "note": "A rough, scaly patch caused by sun damage. It can progress to squamous cell carcinoma."},
    "bcc": {"name": "Basal cell carcinoma", "risk": "Malignant",
            "note": "The most common skin cancer. It grows slowly and rarely spreads, but it needs treatment."},
    "bkl": {"name": "Benign keratosis-like lesion", "risk": "Benign",
            "note": "Includes seborrheic keratosis and solar lentigo. Common and non-cancerous."},
    "df": {"name": "Dermatofibroma", "risk": "Benign",
           "note": "A firm, harmless nodule in the skin."},
    "mel": {"name": "Melanoma", "risk": "Malignant",
            "note": "The most dangerous skin cancer. Finding it early greatly improves outcomes."},
    "nv": {"name": "Melanocytic nevus", "risk": "Benign",
           "note": "A common mole. Usually harmless, but a mole that changes should be checked."},
    "vasc": {"name": "Vascular lesion", "risk": "Benign",
             "note": "A blood-vessel lesion such as an angioma. Usually harmless."},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DermaNet(nn.Module):
    """EfficientNet-B0 backbone + dropout + linear classifier.

    forward() is split in two so Grad-CAM can read the last convolutional
    feature map directly (no hooks needed).
    """

    def __init__(self, num_classes=len(CLASS_CODES), pretrained=True, dropout=0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b0(weights=weights).features  # -> (B, 1280, H/32, W/32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))

    def forward_features(self, x):
        return self.features(x)

    def forward_head(self, feats):
        return self.head(torch.flatten(self.pool(feats), 1))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def eval_transform(img_size=224):
    """Preprocessing used for validation, testing and the app (must match training)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(path, device="cpu"):
    """Return (model, class_names, img_size) from a checkpoint written by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DermaNet(len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["class_names"], ckpt["img_size"]


def predict(model, x):
    """Class probabilities (numpy array) for one preprocessed image tensor [1, 3, H, W]."""
    model.eval()
    with torch.no_grad():
        return model(x).softmax(dim=1)[0].cpu().numpy()


def grad_cam(model, x, class_idx=None):
    """Grad-CAM (Selvaraju et al., 2017) for one image tensor x of shape [1, 3, H, W].

    1. Forward pass up to the last convolutional feature maps A (K channels).
    2. Back-propagate the score of the chosen class to A.
    3. Channel weights = global average of those gradients.
    4. CAM = ReLU(sum_k weight_k * A_k), up-sampled to the input size, scaled to 0-1.

    Returns (cam [H, W] numpy, class probabilities numpy, index of the class explained).
    """
    model.eval()
    with torch.enable_grad():
        feats = model.forward_features(x)
        if not feats.requires_grad:  # e.g. when the parameters are frozen
            feats.requires_grad_(True)
        logits = model.forward_head(feats)
        probs = logits.softmax(dim=1)[0].detach().cpu().numpy()
        if class_idx is None:
            class_idx = int(probs.argmax())
        grads = torch.autograd.grad(logits[0, class_idx], feats)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats.detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), probs, class_idx


def overlay_heatmap(image, cam, alpha=0.6):
    """Colourise a 0-1 CAM and blend it onto a PIL image.

    Returns (heatmap, overlay) as PIL images at the original image size.
    """
    image = image.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
    c = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = (colormaps["jet"](c)[..., :3] * 255).astype(np.uint8)
    w = (alpha * c)[..., None]  # more heat where the model's attention is high
    out = (1 - w) * np.asarray(image, dtype=np.float32) + w * heat
    return Image.fromarray(heat), Image.fromarray(out.astype(np.uint8))
