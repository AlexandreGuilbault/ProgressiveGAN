import torch
from torch import nn
from torch import utils
import torch.optim as optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np

from Models import Discriminator, Generator, init_weights_normal

import time

def calculate_loss(d_out, device='cpu', real=True, smoothing=False):
    # With Label smoothing : Salimans et. al. 2016
    # With Least Squares Loss : Mao et. al. 2016
    criterion = nn.MSELoss()
    
    if real:
        if smoothing : loss = criterion(d_out.squeeze(), (torch.rand(d_out.size(0))/2+0.7).to(device))
        else : loss = criterion(d_out.squeeze(), torch.ones(d_out.size(0)).to(device))
    else:
        if smoothing : loss = criterion(d_out.squeeze(), (torch.rand(d_out.size(0))*0.3).to(device))
        else : loss = criterion(d_out.squeeze(), torch.zeros(d_out.size(0)).to(device))    
        
    return loss

class Coach():
    def __init__(self, input_dimension_size, num_train_images, data_dir, device):
        
        self.device = device
        self.input_dimension_size = input_dimension_size

        self.generator = Generator(self.input_dimension_size).to(self.device)
        self.discriminator = Discriminator(self.input_dimension_size).to(self.device)
        
        self.generator.apply(init_weights_normal)
        self.discriminator.apply(init_weights_normal)
        
        
        
        self.data_dir = data_dir
        self.num_train_images = num_train_images
        
        self.level = 1
        self.samples = {}
        
        # Better to sample from a gaussian distribution
        self.fixed_z = torch.randn(16, 25*self.input_dimension_size**2).to(self.device) 
    
    def set_level(self, level, d_lr, g_lr, beta_l, beta_h, batch_size):
        self.level = level
        
        self.discriminator.set_level(level)
        self.generator.set_level(level)   
        
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)
        
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=d_lr)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(beta_l, beta_h))
        
        self.set_dataloader((2**(level-1))*self.input_dimension_size*2, batch_size)
        

    def set_dataloader(self, size, batch_size):
        self.batch_size = batch_size
        
        transform = transforms.Compose([transforms.CenterCrop(150), transforms.Resize(size), transforms.ToTensor()])
        
        dataset = datasets.ImageFolder(self.data_dir, transform)
        self.dataloader = utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=utils.data.SubsetRandomSampler([*range(self.num_train_images)]), num_workers=0)
    
    def train(self, n_epochs):
        
        samples = []
        
        # Use SGD for discriminator and ADAM for generator, See Radford et. al. 2015
        start_time = time.time()
        
        num_batches_per_epoch = self.num_train_images/self.batch_size
        total_num_batches = n_epochs*num_batches_per_epoch
        
        
        losses = []
        for epoch in range(1, n_epochs+1):

            self.discriminator.train()
            self.generator.train()
            
            for i, data in enumerate(self.dataloader,1):
                
                real_images, _ = data
                real_images = real_images.to(self.device)
                real_images = real_images*2-1  # Rescale input images from [0,1] to [-1, 1]
        
                z = torch.randn(self.batch_size, 25*self.input_dimension_size**2).to(self.device) # Better to sample from a gaussian distribution
        
                # Generate fake images
                fake_images = self.generator(z)
                
                #################################
                # Train Discriminator
                
                self.d_optimizer.zero_grad()
                
                # Discriminator with real images
                d_real = self.discriminator(real_images)
                r_loss = calculate_loss(d_real, device=self.device, real=True, smoothing=True)
        
                # Discriminator with fake images
                d_fake = self.discriminator(fake_images)
                f_loss = calculate_loss(d_fake, device=self.device, real=False, smoothing=True)
                
                d_loss = r_loss + f_loss
        
                # Optimize Discriminator
                d_loss.backward()
                self.d_optimizer.step()
                
                #################################
                # Train Generator  
                
                self.g_optimizer.zero_grad()
                fake_images = self.generator(z)
                d_fake = self.discriminator(fake_images)
                g_loss = calculate_loss(d_fake, device=self.device, real=True, smoothing=True)
        
                # Optimize Generator
                g_loss.backward()
                self.g_optimizer.step()       
                
                # Print losses
                if i % (num_batches_per_epoch//5) == 0:
                    completion = (((epoch-1)*num_batches_per_epoch) + i)/total_num_batches
                    elapsed_time = time.time() - start_time
                    remaining_time = elapsed_time * (1/completion - 1)
                    
                    em, es = divmod(elapsed_time, 60)
                    eh, em = divmod(em, 60)
                    
                    rm, rs = divmod(remaining_time, 60)
                    rh, rm = divmod(rm, 60)
                    
                    print('Epoch {:2d}/{} | Batch_id {:4d}/{:.0f} | d_loss: {:.4f} | g_loss: {:.4f} | Elapsed time : {:.0f}h {:02.0f}min {:02.0f}sec | Remaining time : {:.0f}h {:02.0f}min {:02.0f}sec'.format(epoch, n_epochs, i, num_batches_per_epoch, d_loss.item(), g_loss.item(), eh, em, es, rh, rm, rs))
        
            # Generate and save samples of fake images
            losses.append((d_loss.item(), g_loss.item()))
        
            self.generator.eval() 
            fake_images = self.generator(self.fixed_z)
            samples.append(fake_images.detach()[0:20])
        
        self.samples[self.level] = samples
        
        #######################
        # Print training losses
        fig, ax = plt.subplots()
        losses = np.array(losses)
        plt.plot(losses.T[0], label='Discriminator')
        plt.plot(losses.T[1], label='Generator')
        plt.title("GAN Training Losses")
        plt.legend()
        
    def print_results_per_epoch(self, level=1, step=10):
        samples_arr = np.stack(self.samples[level], axis=0) # shape = [Epoch, Batch Size, Color Channel, Width, Height]
        
        for epoch in range(0, samples_arr.shape[0], step):
            fig, axes = plt.subplots(figsize=(15,2), nrows=1, ncols=7, sharey=True, sharex=True)
            for ax, img in zip(axes.flatten(), samples_arr[epoch, 0:7, :, :, :]):
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.imshow(np.transpose(np.clip(((img+1)/2), a_min=0.,a_max=1.), (1, 2, 0)))
            fig.suptitle("Epoch : {}".format(epoch+1))
            
    def generate_random_image(self, level = 1):
        
        self.generator.set_level(level)
        self.discriminator.set_level(level) # For consistency
        
        imgs = self.generator(torch.randn(1, 25*self.input_dimension_size**2).to(self.device))
        img = imgs.detach()[0]
        plt.imshow(np.transpose((img+1)/2, (1, 2, 0)))

    def print_random_image(self):
        
        dataiter = iter(self.dataloader)
        images, _ = dataiter.next()
        
        image = images[0]
        plt.imshow(np.transpose(image, (1,2,0)))

    def save(self, path='.\\model\\'):
        torch.save(self.generator.state_dict(), path+"Generator_level_{}.pt".format(self.level))
        torch.save(self.discriminator.state_dict(), path+"Discriminator_level_{}.pt".format(self.level))

    def load(self, level, path='.\\model\\'):
        self.generator.load_state_dict(torch.load(path+"Generator_level_{}.pt".format(level)))
        self.discriminator.load_state_dict(torch.load(path+"Discriminator_level_{}.pt".format(level)))
        