"""
Professor K - CNN Model for Voice Impersonation Detection
Input : Mel-spectrogram image (1 x 128 x 128)
Output: 2 classes -> [Real Human, AI-Generated]

Model Name: Professor K
"""
import torch
import torch.nn as nn

MODEL_NAME = 'Professor K'


class VoiceCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        # ---- Feature Extraction Blocks (4 conv blocks) ----
        self.features = nn.Sequential(
            # Block 1: (1, 128, 128) -> (32, 64, 64)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # ---- Classification Head ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 256 * 8 * 8 = 16384
            nn.Dropout(dropout),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits (CrossEntropyLoss applies softmax internally)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = VoiceCNN()
    print(model)
    print(f'\nTrainable parameters: {count_parameters(model):,}')

    # quick shape test
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f'Input shape : {tuple(dummy.shape)}')
    print(f'Output shape: {tuple(out.shape)}  (4 samples, 2 classes)')
    probs = torch.softmax(out, dim=1)
    print(f'Softmax sums: {probs.sum(dim=1)}')
