
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image

class CustomCIFAR10(datasets.CIFAR10):

    def apply_augmentations(self, image):
        augmented_image = self.transform(image=image)  # Pass 'image' as named argument
        return augmented_image

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def compose_custom_transforms(transforms_collection):
        return A.Compose(transforms_collection)

def create_basic_transforms_collection(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)):
    transforms_collection  = [ A.Normalize(mean=mean, std=std),
                               ToTensorV2()]
    
    return transforms_collection

def create_shift_scale_rotate_transform(shift_limit = 0.1, scale_limit = 0.1, rotate_limit = 10):
    return A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, p=0.25)

def create_coarse_drop_out_transformation(max_height=8, max_width=8, fill_value = [0,0,0]):
     return A.CoarseDropout(max_holes=1, min_holes=1, 
                            max_height=max_height, max_width=max_width,
                            min_height=max_height, min_width=max_width, p=0.5, fill_value=fill_value)


def create_coarse_drop_out_transformation(max_height=8, max_width=8, fill_value = [0,0,0]):
     return A.CoarseDropout(max_holes=1, min_holes=1, 
                            max_height=max_height, max_width=max_width,
                            min_height=max_height, min_width=max_width, p=0.5, fill_value=fill_value)


def create_random_resize_crop_transformation():
    return [A.PadIfNeeded(min_height=40, min_width=40,always_apply=True),     
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.Resize(height=32, width=32, always_apply=True)]

def create_flip_transformation(is_horizontal = True, is_random = False):
    if(is_random):
        return A.Flip(p=0.25)
    if(is_horizontal):
        return A.HorizontalFlip( p=0.25)
    else:
         return A.VerticalFlip(p=0.25)

def get_CIFAR10_datasets(train_transforms_collection, test_transforms_collection, data_folder) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:

    train_dataset = CustomCIFAR10( root=data_folder,
                                    train=True,
                                    download=True,
                                    transform=train_transforms_collection)
    
    test_dataset = CustomCIFAR10( root=data_folder,
                                        train=False,
                                        download=True,
                                        transform=test_transforms_collection)
    
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    for batch_data, label in test_dataloader:
    # (e.g., shape: (batch_size, 1 channel, 28, 28)). (batch_size, channels, height, width)
    # y would contain the corresponding labels for each image, indicating the actual digit represented in the image 
        print(f"Shape of test_dataloader batch_data [Batch, C, H, W]: {batch_data.shape}")
        print(f"Shape of test_dataloader label (label): {label.shape} {label.dtype}")
        print(f"Labels for a batch of size {batch_size} are {label}")
        break

    return train_dataloader, test_dataloader