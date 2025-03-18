import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

# Constants
BATCH_SIZE = 32
DATASET_PATH = "./datasets/"

class CustomRotation:
    """Rotates images to correct orientation."""
    def __call__(self, image):
        return image.permute(2, 0, 1)  # Equivalent to image.transpose(0, 2).transpose(0, 1)

def get_transform(train: bool):
    """Prepares a transformation pipeline with normalization & resizing."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def make_image_transform(image_transform_params: dict, transform=None):
    """Creates an image transformation pipeline based on input parameters."""
    resize_modes = {
        "shrink": transforms.Resize,
        "crop": transforms.CenterCrop
    }
    resize_op = resize_modes.get(image_transform_params['image_mode'], None)

    if resize_op:
        preprocess_image = resize_op((image_transform_params['output_image_size']['width'],
                                      image_transform_params['output_image_size']['height']))
        return transforms.Compose([preprocess_image, transform]) if transform else preprocess_image
    return transform

def read_voc_dataset(download=True, year='2007', batch_size=BATCH_SIZE):
    """Loads PASCAL VOC dataset with transformations and returns dataloaders."""
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.VOCDetection(DATASET_PATH, year=year, image_set='train',
                                                      download=download, transform=transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = torchvision.datasets.VOCDetection(DATASET_PATH, year=year, image_set='val',
                                                    download=download, transform=transform_pipeline)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # return train_loader, val_loader
    return train_dataset, val_dataset

def get_images_labels(dataloader):
    """Extracts images and labels from a dataloader batch."""
    return next(iter(dataloader))

class NoisySBDataset:
    """Handles SB dataset loading with optional transforms."""
    def __init__(self, path, image_set="train", transforms=None, download=True):
        self.transforms = transforms
        self.dataset = torchvision.datasets.SBDataset(root=path, image_set=image_set, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, truth = self.dataset[idx]
        return (self.transforms(img) if self.transforms else img, truth)

def read_sbd_dataset(batch_size=BATCH_SIZE, download=True):
    """Loads and normalizes SB dataset."""
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = NoisySBDataset(DATASET_PATH, image_set='train', download=download, transforms=transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = NoisySBDataset(DATASET_PATH, image_set='val', download=download, transforms=transform_pipeline)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
