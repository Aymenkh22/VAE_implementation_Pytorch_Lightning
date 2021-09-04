from pytorch_lightning import trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
import torchvision
from torchvision import datasets, transforms
import numpy as np



class VAE(pl.LightningModule):
    def __init__(self, image_size =32, channel_num =3, kernel_num = 256, z_size = 128):
            # configurations
            super().__init__()
            self.image_size = image_size
            self.channel_num = channel_num
            self.kernel_num = kernel_num
            self.z_size = z_size

            # encoder
            self.encoder = nn.Sequential(
                self._conv(channel_num, kernel_num // 8, kernel_size=4, stride=2, padding=1),
                self._conv(kernel_num // 8, kernel_num // 4, kernel_size=4, stride=2, padding=1),
                self._conv(kernel_num // 4, kernel_num // 2, kernel_size=4, stride=2, padding=1),
                self._conv(kernel_num // 2, kernel_num, kernel_size=3, stride=1, padding=1),
            )

            # encoded feature's size and volume
            self.feature_size = image_size // 8
            self.feature_volume = kernel_num * (self.feature_size ** 2)

            # q
            self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
            
            self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

            # projection
            self.project = self._linear(z_size, self.feature_volume, relu=False)

            # decoder
            self.decoder = nn.Sequential(
                self._deconv(kernel_num, kernel_num // 2, kernel_size=4, stride=2, padding=1),
                self._deconv(kernel_num // 2, kernel_num // 4, kernel_size=4, stride=2, padding=1),
                self._deconv(kernel_num // 4, kernel_num // 8, kernel_size=4, stride=2, padding=1),
                self._deconv(kernel_num // 8, channel_num, kernel_size=3, stride=1, padding=1, last = True)
            )
    def forward(self, x):
                # encode x
                encoded = self.encoder(x)

                # sample latent code z from q given x.
                unrolled = encoded.view(-1, self.feature_volume)
                mean= self.q_mean(unrolled)
                logvar =0
                
                logvar=self.q_logvar(unrolled)

                z = self.z(mean, logvar)
               
                z_projected = self.project(z).view(
                    -1, self.kernel_num,
                    self.feature_size,
                    self.feature_size,
                )

                x_reconstructed = self.decoder(z_projected)

                return x_reconstructed, z, mean, logvar

            # ==============
            # VAE components
            # ==============



    def z(self, mean, logvar):
                #std = logvar.mul(0.5).exp_()
                std=torch.exp(0.5*logvar)
                #eps = (Variable(torch.randn(std.size())))
                eps=torch.randn_like(std)
                return eps*std+mean

    def _conv(self, channel_size, kernel_num, kernel_size=4, stride=2, padding=1):
                return nn.Sequential(
                    nn.Conv2d(
                        channel_size, kernel_num,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                    ),
                    nn.BatchNorm2d(kernel_num),
                    nn.ReLU(),
                )

    def _deconv(self, channel_num, kernel_num, kernel_size=4, stride=2, padding=1,last = False):
                seq = [nn.ConvTranspose2d(
                        channel_num, kernel_num,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                    )]
                if last == False:
                    seq.append(nn.BatchNorm2d(kernel_num))
                    seq.append(nn.ReLU())
                return nn.Sequential(*seq)

    def _linear(self, in_size, out_size, relu=True):
                return nn.Sequential(
                    nn.Linear(in_size, out_size),
                    nn.ReLU(),
                ) if relu else nn.Linear(in_size, out_size)
    
    def loss_function(self,recon_x,x,mu,logvar):
            BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='sum')

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
    
        
    def validation_step(self, batch,batch_idx):

        x,y=batch
        recon_batch, z, mu, logvar=self(x)
        val_loss=self.loss_function(recon_batch,x,mu,logvar)
         
        return {'val_loss':val_loss,'x_hat':recon_batch,'val_latent_vector':z ,'val_labels':y}
        
    def training_step(self, batch,batch_idx):
        
            x,y=batch[0]
            recon_batch, z, mu, logvar=self(x)
            loss=self.loss_function(recon_batch,x,mu,logvar)
  
        


            return{'loss':loss,'train_latent_vec':z,'train_labels':y}

    def configure_optimizers(self):
        
        return torch.optim.Adam(self.parameters(),lr=1e-3)

    def train_dataloader(self):

        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.201])])
        train_data = datasets.CIFAR10(root='data', train=True,download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.201])])
        test_data = datasets.CIFAR10(root='data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

        return [train_loader,test_loader]

    def val_dataloader(self):

        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.201])])
        test_data = datasets.CIFAR10(root='data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
 
        return test_loader


if __name__ == '__main__':

    num_workers = 0

    batch_size = 64

    model=VAE()

    trainer=pl.Trainer(fast_dev_run=True,max_epochs=2)#max_epochs : to define the maximum number of training epochs

    #To test if the model works correctly you can use pl.Trainer(fast_dev_run=True)

    trainer.fit(model)
     
    #To visit the tensorboard press in the console :tensorboard --logdir lightning_logs 
    # then copy the link provided into the browser 
     
    
