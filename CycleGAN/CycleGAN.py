import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from utils import load_checkpoint, save_checkpoint, seed_everything
import config

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias = True, padding_mode = "reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self,in_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride = 2,
                padding = 1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )
        in_channels = features[0]
        layers = []
        for feature in features[1:]:
            layers.append(Block(in_channels,feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature

        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
    
class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, down=True, use_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs)
            if down
            else nn.ConvTranspose2d(in_channels,out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()

        )
    def forward(self,x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Block = nn.Sequential(
            ConvBlock(channels,channels,kernel_size=3,padding=1),
            ConvBlock(channels,channels,use_act=False,kernel_size=3,padding=1),
        )

    def forward(self,x):
        return x + self.Block(x)
    
class Generator(nn.Module):
    def __init__(self, img_channels,num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3,padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.down_block = nn.ModuleList(
            [
                ConvBlock(num_features,num_features*2,kernel_size=3,stride=2,padding=1),
                ConvBlock(num_features*2,num_features*4,kernel_size=3,stride=2,padding=1),
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_block = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2,padding=1,output_padding=1),
                ConvBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2,padding=1,output_padding=1),
            ]
        )
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1,padding=3,padding_mode="reflect")

    def forward(self,x):
        x = self.initial(x)
        for layer in self.down_block:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_block:
            x = layer(x)

        return torch.tanh(self.last(x))
    
class CycleGANDataset(Dataset):
    def __init__(self,root_A, root_B, transforms=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transforms = transforms

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images),len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len =len(self.B_images)

    def __len__(self):
        return self.length_dataset
    def __getitem__(self,index):
        A = self.A_images[index % self.A_len]
        B = self.B_images[index % self.B_len]

        A_path = os.path.join(self.root_A,A)
        B_path = os.path.join(self.root_B,B)

        A = np.array(Image.open(A_path).convert("RGB"))
        B = np.array(Image.open(B_path).convert("RGB"))
        
        if self.transforms:
            A = self.transforms(image=A)["image"]
            B = self.transforms(image=B)["image"]

        return A, B
    
def train_fn(disc_A,disc_B,gen_A,gen_B,loader,opt_disc,opt_gen,l1,mse):
    loop =tqdm(loader,leave=True)

    for index, (A,B) in enumerate(loop):
        A = A.to(config.DEVICE)
        B = B.to(config.DEVICE)
        
        fake_A = gen_A(B)
        D_A_real = disc_A(A)
        D_A_fake = disc_A(fake_A.detach())
        D_A_real_loss = mse(D_A_real,torch.ones_like(D_A_real))
        D_A_fake_loss = mse(D_A_fake,torch.zeros_like(D_A_fake))
        D_A_loss = (D_A_real_loss + D_A_fake_loss)
        
        fake_B = gen_B(A)
        D_B_real = disc_B(B)
        D_B_fake = disc_B(fake_B.detach())
        D_B_real_loss = mse(D_B_real,torch.ones_like(D_B_real))
        D_B_fake_loss = mse(D_B_fake,torch.zeros_like(D_B_fake))
        D_B_loss = (D_B_real_loss + D_B_fake_loss)

        D_loss = (D_A_loss + D_B_loss)/2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        D_A_fake = disc_A(fake_A)
        D_B_fake = disc_B(fake_B)

        loss_G_A = mse(D_A_fake,torch.ones_like(D_A_fake))
        loss_G_B = mse(D_B_fake,torch.ones_like(D_B_fake))

        cycle_B = gen_B(fake_A)
        cycle_A = gen_A(fake_B)
        cycle_loss_A = l1(A,cycle_A)
        cycle_loss_B = l1(B,cycle_B)

        '''identity_A = gen_A(A)
        identity_B = gen_B(B)
        identity_loss_A = l1(A,identity_A)
        identity_loss_B = l1(B,identity_B)'''

        G_loss = loss_G_A +loss_G_B + (cycle_loss_A + cycle_loss_B)*config.LAMBDA_CYCLE
        ''' + (identity_loss_A + identity_loss_B)*config.LAMBDA_IDENTITY '''

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if index % 200 == 0:
            save_image(fake_A*0.5 + 0.5, f"results/A/fake_A_{index}.png")
            save_image(fake_B*0.5 + 0.5, f"results/B/fake_B_{index}.png")

def main():
    print(f"Device: {config.DEVICE}")
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_A = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    gen_B = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999)
    )
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE
        )

    dataset = CycleGANDataset(
        root_A = "dataset/train/trainA", root_B = "dataset/train/trainB", transforms = config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle= True,

    )

    for epoch in range(config.NUM_EPOCHS):
        print(f"{epoch}/{config.NUM_EPOCHS}")
        train_fn(disc_A,disc_B,gen_A,gen_B,loader,opt_disc,opt_gen,l1,mse)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A,opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_B,opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_A,opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_B,opt_disc, filename=config.CHECKPOINT_CRITIC_B)

def test_disc():
    x = torch.randn((1,3,256,256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

def test_generator():
    x = torch.randn((1,3,256,256))
    gen = Generator(3,9)
    print(gen(x).shape)

if __name__ == "__main__":
    main()