# Wasserstein GAN
"""
-PROS
--Stable
-CONS
--Longer Training Time
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty

class Critic(nn.Module):
    def __init__(self,channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            #Input image: N x Channel_image x 64 x 64
            nn.Conv2d(
                channels_img,features_d,kernel_size=4,stride=2,padding=1,
            ),#32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),#16 x 16
            self._block(features_d*2,features_d*4,4,2,1),#8 x 8
            self._block(features_d*4,features_d*8,4,2,1),#4 x 4
            nn.Conv2d(features_d*8,1,kernel_size=4,stride = 2, padding = 0),# 1 x 1
        )
    def _block(self,in_channels,out_channels, kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size,stride,padding,bias=False),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            #Input: N x z_dim x 1 x 1
            self._block(z_dim,features_g*16,4,1,0),#Nxf_g*16x4x4
            self._block(features_g*16,features_g*8,4,2,1),#8x8
            self._block(features_g*8, features_g*4,4,2,1),#16x16
            self._block(features_g*4, features_g*2,4,2,1),#32x32
            nn.ConvTranspose2d(
                features_g*2,channels_img,kernel_size=4,stride=2,padding = 1
            ),
            nn.Tanh(),
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight.data,0.0,0.02)

        if isinstance(m,nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data,0.0,0.02)
        
        if isinstance(m,nn.BatchNorm2d):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N,in_channels,H,W = 8,3,64,64
    z_dim = 100
    x = torch.randn((N,in_channels,H,W))
    critic = Critic(in_channels,8)
    initialize_weights(critic)
    assert critic(x).shape == (N,1,1,1)
    gen = Generator(z_dim,in_channels,8)
    initialize_weights(gen)
    z=torch.randn((N,z_dim,1,1))
    assert gen(z).shape == (N,in_channels,H,W)
    print("test_success")

device = "xpu" if torch.xpu.is_available() else "cpu"
print(f"Device: {device}")
lr = 1e-4
batch_size = 64
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10



transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],
            [0.5 for _ in range(channels_img)]
            )
    ]
)

dataset = datasets.MNIST(root="dataset", train=True, transform = transforms, download = True)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

gen = Generator(z_dim,channels_img,features_gen).to(device)
critic = Critic(channels_img,features_disc).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr = lr, betas = (0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas = (0.0,0.9))

fixed_noise = torch.randn(32,z_dim,1,1).to(device)

writer_real = SummaryWriter(f"runs/WGAN/real")
writer_fake = SummaryWriter(f"runs/WGAN/fake")

step = 0
gen.train()
critic.train()

if device == "xpu":
    torch.xpu.synchronize()

for epoch in range(num_epochs):
    for batch_idx, (real,_) in enumerate(dataloader):
        real = real.to(device)

        start =time.time()

        for _ in range(CRITIC_ITERATIONS):
            curr_batch_size = real.shape[0]
            noise = torch.randn((curr_batch_size,z_dim,1,1)).to(device)
            fake = gen(noise).to(device)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph = True)
            opt_critic.step()

        ### Train the generator
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        end = time.time()
        if batch_idx % 100 == 0:
            print(f"Epoch[{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \nLoss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}\nTime Take: {-(start-end)}")

        with torch.no_grad():
            fake = gen(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(
                real[:32],normalize=True
            )
            img_grid_fake = torchvision.utils.make_grid(
                fake[:32],normalize=True
            )

            writer_real.add_image("WGAN_Real",img_grid_real,global_step = step)
            writer_fake.add_image("WGAN_Fake",img_grid_fake,global_step = step)
        step+=1
