import torch
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(torch.nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        blocks = []
        input_channels = 3
        for output_channels in (32, 64, 128):
            blocks.extend([
                torch.nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(output_channels), torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            ])
            input_channels = output_channels
        self.features = torch.nn.Sequential(*blocks, torch.nn.AdaptiveAvgPool2d(1))
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(128, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(architecture: str, classes: int, pretrained: bool = False):
    if architecture == "small_cnn":
        return SmallCNN(classes)
    if architecture != "resnet18":
        raise ValueError(f"Unknown architecture: {architecture}")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(model.fc.in_features, classes))
    return model


def configure_resnet_stage(model, fine_tuning: bool):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if fine_tuning:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def set_training_mode(model, architecture: str, fine_tuning: bool):
    if architecture == "small_cnn":
        model.train()
    else:
        model.eval()  # Preserve frozen backbone BatchNorm statistics.
        model.fc.train()
        if fine_tuning:
            model.layer4.train()
