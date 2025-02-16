#!/bin/env python3

import argparse
import os
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import EuroSAT, ImageFiles, random_split
from collections import namedtuple

# Named tuple to store test results
TestResult = namedtuple('TestResult', 'truth predictions')

# Global variables for Grad-CAM hook
feature_maps = None
gradients = None

def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def register_hooks(model):
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # Find the last convolutional layer
            last_conv_layer = module
    if last_conv_layer:
        last_conv_layer.register_forward_hook(forward_hook)
        last_conv_layer.register_full_backward_hook(backward_hook)

def apply_colormap_on_image(org_im, activation):
    activation = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
    activation = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(activation, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(org_im, 0.6, heatmap, 0.4, 0)
    return superimposed_img

def generate_gradcam(model, image_tensor, class_idx):
    global feature_maps, gradients
    model.zero_grad()
    image_tensor.requires_grad_()
    
    with torch.set_grad_enabled(True):
        output = model(image_tensor)
        score = output[:, class_idx]
        score.backward()
    
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    weights = weights.view(1, -1, 1, 1)  # Ensure matching shape with feature_maps
    
    cam = torch.sum(weights * feature_maps, dim=1).squeeze().cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    return cam

def visualize_gradcam(image_path, heatmap, save_path):
    org_img = cv2.imread(image_path)
    if org_img is None:
        print(f"Warning: Unable to load image {image_path}")
        return
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    superimposed_img = apply_colormap_on_image(org_img, heatmap)
    cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM visualization at {save_path}")

@torch.no_grad()
def predict(model, dl, show_progress=True, gradcam=False):
    if show_progress:
        dl = tqdm(dl, "Predict", unit="batch")
    device = next(model.parameters()).device
    model.eval()
    preds = []
    truth = []
    i = 0
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        p = outputs.argmax(1).tolist()
        preds += p
        truth += labels.tolist()
        if gradcam:
            for img, pred in zip(images, p):
                heatmap = generate_gradcam(model, img.unsqueeze(0), pred)
                save_path = f"gradcam_{i}.jpg"
                visualize_gradcam(f"image_{i}.jpg", heatmap, save_path)
                i += 1
    return TestResult(truth=torch.as_tensor(truth), predictions=torch.as_tensor(preds))

def report(result, label_names):
    from sklearn.metrics import classification_report, confusion_matrix
    cr = classification_report(result.truth, result.predictions, target_names=label_names, digits=3)
    confusion = confusion_matrix(result.truth, result.predictions)
    try:
        import pandas as pd
        confusion = pd.DataFrame(confusion, index=label_names, columns=[s[:3] for s in label_names])
    except ImportError:
        pass
    print("Classification report")
    print(cr)
    print("Confusion matrix")
    print(confusion)

def main(args):
    save = torch.load(args.model, map_location='cpu')
    normalization = save['normalization']
    model = models.resnet50(num_classes=save['model_state']['fc.bias'].numel())
    model.load_state_dict(save['model_state'])
    model = model.to(args.device)
    register_hooks(model)
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**normalization)])
    dataset = EuroSAT(transform=tr)
    trainval, test = random_split(dataset, 0.9, random_state=42)
    test_dl = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    result = predict(model, test_dl, gradcam=args.gradcam)
    report(result, dataset.classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict labels and generate Grad-CAM.")
    parser.add_argument('-m', '--model', default='weights/best.pt', type=str, help="Model for prediction")
    parser.add_argument('-j', '--workers', default=4, type=int, help="Number of DataLoader workers")
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--gradcam', action='store_true', help="Enable Grad-CAM visualization")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
