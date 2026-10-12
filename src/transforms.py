from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_transform(training: bool = False):
    if training:
        spatial = [
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ]
    else:
        spatial = [transforms.Resize(256), transforms.CenterCrop(224)]
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
