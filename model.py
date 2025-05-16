import torchvision
import torch.nn as nn
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.ssd import (
    SSD,
    DefaultBoxGenerator,
    SSDHead
)

def create_model(num_classes=91, size=300, nms=0.45):
    # Load pretrained ResNet50 model
    model_backbone = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    )

    # Choose intermediate layers for SSD multi-scale feature maps
    return_nodes = {
        'layer2': '0',  # 1/8 scale
        'layer3': '1',  # 1/16 scale
        'layer4': '2',  # 1/32 scale
    }

    # Extract backbone features
    backbone = create_feature_extractor(model_backbone, return_nodes=return_nodes)

    # Extra layers to produce 6 total feature maps
    extra_layers = nn.ModuleList([
    nn.Sequential(  # From 2048 ➜ 512
        nn.Conv2d(2048, 512, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    ),
    nn.Sequential(  # From 512 ➜ 256
        nn.Conv2d(512, 256, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    ),
    nn.Sequential(  # From 256 ➜ 256
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
        nn.ReLU()
    )
    ])

# Corrected output channel sizes after each feature map
    out_channels = [512, 1024, 2048, 512, 256, 256]

    # Combine backbone + extra layers
    class SSDBackbone(nn.Module):
        def __init__(self, backbone, extras):
            super().__init__()
            self.backbone = backbone
            self.extras = extras

        def forward(self, x):
            x = self.backbone(x)  # dict or list
            if isinstance(x, dict):
                features = list(x.values())
            elif isinstance(x, (list, tuple)):
                features = list(x)
            else:
                raise TypeError(f"Expected dict/list, got {type(x)}")

            # Apply extra SSD layers
            x = features[-1]
            for layer in self.extras:
                x = layer(x)
                features.append(x)
            # for i, f in enumerate(features):
            #   print(f"Feature {i}: shape = {f.shape}")

    # return{str(i): f for i, f in enumerate(features)}
            return {str(i): f for i, f in enumerate(features)}

    full_backbone = SSDBackbone(backbone, extra_layers)

    # Anchor generator: 6 scales
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    # Detection head
    head = SSDHead(out_channels, num_anchors, num_classes)

    # Final SSD model
    model = SSD(
        backbone=full_backbone,
        anchor_generator=anchor_generator,
        size=(size, size),
        num_classes=num_classes,
        head=head,
        nms_thresh=nms
    )

    return model

if __name__ == '__main__':
    model = create_model(num_classes=2, size=300)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
