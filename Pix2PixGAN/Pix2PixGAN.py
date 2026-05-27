import os
import config
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import save_checkpoint, load_checkpoint, save_some_examples

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = stride, bias=False, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features = [64,128,256,512]): #256 -> 30x30
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0],kernel_size = 4, stride = 2, padding = 1, padding_mode = "reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect")
        )
        self.model = nn.Sequential(
            *layers
        )

    def forward(self,x,y):
        x = torch.cat([x,y], dim = 1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,bias=False,padding_mode="reflect")
            if down 
            else nn.ConvTranspose2d(in_channels,out_channels,4,2,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2)   
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features,4,2,1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down1 = Block(features, features*2, down = True, act = "leaky",use_dropout=False)
        self.down2 = Block(features*2, features*4, down = True, act = "leaky",use_dropout=False)
        self.down3 = Block(features*4, features*8, down = True, act = "leaky",use_dropout=False)
        self.down4 = Block(features*8, features*8, down = True, act = "leaky",use_dropout=False)
        self.down5 = Block(features*8, features*8, down = True, act = "leaky",use_dropout=False)
        self.down6 = Block(features*8, features*8, down = True, act = "leaky",use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode="reflect"),nn.ReLU(),
        )
        self.up1 = Block(features*8, features*8, down=False,act="relu",use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False,act="relu",use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False,act="relu",use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False,act="relu",use_dropout=False)
        self.up5 = Block(features*8*2, features*4, down=False,act="relu",use_dropout=False)
        self.up6 = Block(features*4*2, features*2, down=False,act="relu",use_dropout=False)
        self.up7 = Block(features*2*2, features, down=False,act="relu",use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,kernel_size=4,stride=2, padding=1),
            nn.Tanh()
            )
        
    def forward(self,x):
        d1 =self.initial_down(x)
        d2 =self.down1(d1)
        d3 =self.down2(d2)
        d4 =self.down3(d3)
        d5 =self.down4(d4)
        d6 =self.down5(d5)
        d7 =self.down6(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7],1))
        up3 = self.up3(torch.cat([up2,d6],1))
        up4 = self.up4(torch.cat([up3,d5],1))
        up5 = self.up5(torch.cat([up4,d4],1))
        up6 = self.up6(torch.cat([up5,d3],1))
        up7 = self.up7(torch.cat([up6,d2],1))

        final_up = self.final_up(torch.cat([up7,d1],1))
        
        return final_up        

def test_discriminator():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    preds = model(x,y)
    print(preds.shape)

def test_generator():
    x = torch.randn((1,3,256,256))
    model = Generator(in_channels=3, features = 64)
    pred = model(x)
    print(pred.shape)


class Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files= os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)
    def __getitem__(self,index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:,:600,:]
        target_image = image[:,600:,:]

        augmentation = config.both_transform(image = input_image, image0 = target_image)
        input_image,target_image = augmentation["image"], augmentation["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]  

        return input_image, target_image
    
def train(disc, gen, loader, opt_disc, opt_gen, L1_LOSS, BCE):
    disc.train()
    gen.train()
    loop = tqdm(loader,leave=True)
    for idx, (x,y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        y_fake = gen(x)
        d_real = disc(x,y)
        d_fake = disc(x,y_fake.detach())
        d_real_loss = BCE(d_real,torch.ones_like(d_real))
        d_fake_loss = BCE(d_fake,torch.zeros_like(d_fake))
        d_loss =(d_real_loss + d_fake_loss)/2
        
        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()

        d_fake = disc(x,y_fake)
        g_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
        L1 = L1_LOSS(y_fake, y) * config.L1_LAMBDA
        g_loss = g_fake_loss + L1

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

    print(f"LOSS D:{d_loss:.4f}, LOSS G:{g_loss:.4f}")

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(),lr = config.LEARNING_RATE, betas = (0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC,disc, opt_disc, config.LEARNING_RATE)

    train_dataset = Dataset(root_dir="/home/drbigbrain/Desktop/Projects/GANFromScratch/Pix2PixGAN/dataset/pix2pix/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = Dataset(root_dir="/home/drbigbrain/Desktop/Projects/GANFromScratch/Pix2PixGAN/dataset/pix2pix/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range(config.NUM_EPOCHS):
        print(f"{epoch}/{config.NUM_EPOCHS}\n")
        train(disc, gen, train_loader, opt_disc,opt_gen, L1_LOSS, BCE)

        if config.SAVE_MODEL and epoch%5 == 0:
            save_checkpoint(gen,opt_gen,filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)
        save_some_examples(gen, val_loader,epoch, folder="results/Pix2Pix")
if __name__ == "__main__":
    main()
