import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "models/pix2pix/disc.pth.tar"
CHECKPOINT_GEN = "models/pix2pix/gen.pth.tar"

both_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose([
    A.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ToTensorV2(),
])

transform_only_mask = A.Compose([
    A.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ToTensorV2(),
])