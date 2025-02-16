# import os

# import torch
# from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder
# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
# MD5 = "c8fa014336c82ac7804f0398fcb19387"
# SUBDIR = '2750'


# def random_split(dataset, ratio=0.9, random_state=None):
#     if random_state is not None:
#         state = torch.random.get_rng_state()
#         torch.random.manual_seed(random_state)
#     n = int(len(dataset) * ratio)
#     split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
#     if random_state is not None:
#         torch.random.set_rng_state(state)
#     return split


# class EuroSAT(ImageFolder):
#     def __init__(self, root='data', transform=None, target_transform=None):
#         self.download(root)
#         root = os.path.join(root, SUBDIR)
#         super().__init__(root, transform=transform, target_transform=target_transform)

#     @staticmethod
#     def download(root):
#         if not check_integrity(os.path.join(root, "EuroSAT.zip")):
#             download_and_extract_archive(URL, root, md5=MD5)


# # Apparently torchvision doesn't have any loader for this so I made one
# # Advantage compared to without loader: get "for free" transforms, DataLoader
# # (workers), etc.
# class ImageFiles(Dataset):
#     """
#     Generic data loader where all paths must be given
#     """

#     def __init__(self, paths: [str], loader=default_loader, transform=None):
#         self.paths = paths
#         self.loader = loader
#         self.transform = transform

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         image = self.loader(self.paths[idx])
#         if self.transform is not None:
#             image = self.transform(image)
#         # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
#         return image, -1


#!/bin/env python3







# import argparse
# import os
# import cv2
# import numpy as np
# import torch
# import torchvision
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import models, transforms
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# from dataset import EuroSAT, ImageFiles, random_split
# from collections import namedtuple

# # Named tuple to store test results
# TestResult = namedtuple('TestResult', 'truth predictions')

# # Global variables for Grad-CAM hook
# feature_maps = None
# gradients = None


# def forward_hook(module, input, output):
#     """Hook to store feature maps."""
#     global feature_maps
#     feature_maps = output


# def backward_hook(module, grad_in, grad_out):
#     """Hook to store gradients."""
#     global gradients
#     gradients = grad_out[0]


# def register_hooks(model):
#     """Attach hooks to the last convolutional layer."""
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):  # Find the last convolutional layer
#             module.register_forward_hook(forward_hook)
#             module.register_backward_hook(backward_hook)


# def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
#     """Apply heatmap onto original image."""
#     activation = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
#     activation = np.uint8(255 * activation)
#     heatmap = cv2.applyColorMap(activation, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(org_im, 0.6, heatmap, 0.4, 0)
#     return superimposed_img


# def generate_gradcam(model, image_tensor, class_idx):
#     """Generate Grad-CAM heatmap for a given image and class index."""
#     global feature_maps, gradients

#     model.zero_grad()
#     output = model(image_tensor)
#     score = output[:, class_idx]
#     score.backward()

#     # Compute Grad-CAM
#     weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
#     cam = torch.sum(weights * feature_maps, dim=1).squeeze().cpu().detach().numpy()
#     cam = np.maximum(cam, 0)  # ReLU
#     cam = cam / np.max(cam)  # Normalize

#     return cam


# def visualize_gradcam(image_path, heatmap, save_path):
#     """Overlay Grad-CAM heatmap on an image and save the result."""
#     org_img = cv2.imread(image_path)
#     org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

#     # Resize heatmap to match original image size
#     superimposed_img = apply_colormap_on_image(org_img, heatmap)

#     # Save and display the Grad-CAM result
#     cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
#     print(f"Saved Grad-CAM visualization at {save_path}")


# @torch.no_grad()
# def predict(model: nn.Module, dl: torch.utils.data.DataLoader, paths=None, show_progress=True, gradcam=False):
#     """Run model inference and optionally apply Grad-CAM."""
#     if show_progress:
#         dl = tqdm(dl, "Predict", unit="batch")
#     device = next(model.parameters()).device

#     model.eval()
#     preds = []
#     truth = []
#     i = 0

#     for images, labels in dl:
#         images = images.to(device, non_blocking=True)
#         outputs = model(images)
#         p = outputs.argmax(1).tolist()
#         preds += p
#         truth += labels.tolist()

#         if paths and gradcam:
#             for img_path, pred_class in zip(paths, p):
#                 heatmap = generate_gradcam(model, images[i].unsqueeze(0), pred_class)
#                 save_path = f"gradcam_{os.path.basename(img_path)}"
#                 visualize_gradcam(img_path, heatmap, save_path)
#                 i += 1

#         if paths:
#             for pred in p:
#                 print(f"{paths[i]!r}, {pred}")
#                 i += 1

#     return TestResult(truth=torch.as_tensor(truth), predictions=torch.as_tensor(preds))


# def report(result: TestResult, label_names):
#     """Generate classification report and confusion matrix."""
#     from sklearn.metrics import classification_report, confusion_matrix
#     cr = classification_report(result.truth, result.predictions, target_names=label_names, digits=3)
#     confusion = confusion_matrix(result.truth, result.predictions)

#     try:
#         import pandas as pd
#         confusion = pd.DataFrame(confusion, index=label_names, columns=[s[:3] for s in label_names])
#     except ImportError:
#         pass

#     print("Classification report")
#     print(cr)
#     print("Confusion matrix")
#     print(confusion)


# def main(args):
#     """Main function to load model, prepare dataset, and run predictions."""
#     save = torch.load(args.model, map_location='cpu')
#     normalization = save['normalization']
#     model = models.resnet50(num_classes=save['model_state']['fc.bias'].numel())
#     model.load_state_dict(save['model_state'])
#     model = model.to(args.device)

#     # Register Grad-CAM hooks
#     register_hooks(model)

#     tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])
#     if args.files:
#         test = ImageFiles(args.files, transform=tr)
#     else:
#         dataset = EuroSAT(transform=tr)
#         trainval, test = random_split(dataset, 0.9, random_state=42)

#     test_dl = torch.utils.data.DataLoader(
#         test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
#     )
#     result = predict(model, test_dl, paths=args.files, gradcam=args.gradcam)

#     if not args.files:  # Generate report only when running on test set
#         report(result, dataset.classes)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Predict labels and optionally generate Grad-CAM.")
#     parser.add_argument('-m', '--model', default='weights/best.pt', type=str, help="Model for prediction")
#     parser.add_argument('-j', '--workers', default=4, type=int, help="Number of DataLoader workers")
#     parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
#     parser.add_argument('--gradcam', action='store_true', help="Enable Grad-CAM visualization")
#     parser.add_argument('files', nargs='*', help="Files for prediction")
    
#     args = parser.parse_args()
#     args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     main(args)




import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'


def random_split(dataset, ratio=0.9, random_state=None):
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split


class EuroSAT(ImageFolder):
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def download(root):
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)


# Apparently torchvision doesn't have any loader for this so I made one
# Advantage compared to without loader: get "for free" transforms, DataLoader
# (workers), etc.
class ImageFiles(Dataset):
    """
    Generic data loader where all paths must be given
    """

    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        # WARNING -1 indicates no target, it's useful to keep the same interface as torchvision
        return image, -1



