import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# Constants for EuroSAT dataset
URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
MD5 = "c8fa014336c82ac7804f0398fcb19387"
SUBDIR = '2750'


def random_split(dataset, ratio=0.9, random_state=None):
    """Split dataset into training and validation sets while retaining image paths."""
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)

    n = int(len(dataset) * ratio)
    train_set, val_set = torch.utils.data.random_split(dataset, [n, len(dataset) - n])

    if random_state is not None:
        torch.random.set_rng_state(state)

    return train_set, val_set


class EuroSAT(ImageFolder):
    """EuroSAT Dataset Loader with Image Paths."""
    def __init__(self, root='data', transform=None, target_transform=None):
        self.download(root)
        root = os.path.join(root, SUBDIR)
        super().__init__(root, transform=transform, target_transform=target_transform)
    
    @staticmethod
    def download(root):
        """Download and extract EuroSAT dataset if not already present."""
        if not check_integrity(os.path.join(root, "EuroSAT.zip")):
            download_and_extract_archive(URL, root, md5=MD5)

    def __getitem__(self, index):
        """Return image, label, and file path."""
        image, label = super().__getitem__(index)
        path = self.samples[index][0]  # Retrieve file path
        return image, label, path


class ImageFiles(Dataset):
    """Custom dataset loader for direct image files with paths."""
    def __init__(self, paths: [str], loader=default_loader, transform=None):
        self.paths = paths
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """Return image and its file path."""
        image = self.loader(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, -1, self.paths[idx]  # Returning the image, dummy label (-1), and path
