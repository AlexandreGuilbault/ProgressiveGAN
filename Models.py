import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, input_dimension_size):
        super(Generator, self).__init__()

        self.level = 0   
        self.input_dimension_size = input_dimension_size
        
        self.network = nn.Sequential()

        self.set_level(1)
        self.input = nn.Sequential(
                nn.ConvTranspose2d(in_channels=25, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.5),
                )

        self.output = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh())

    def forward(self, x):
        x = self.input(x.view(-1,25, self.input_dimension_size, self.input_dimension_size))
        return self.output(self.network(x))
    
    def set_level(self, level): 
        self.level = level
        
        if "level_{}".format(level) not in self.network._modules.keys():
            self.add_level(level)

    def add_level(self, level):
        
        # convt_output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.network.add_module("level_{}".format(level), nn.Sequential(
                #3xIxI
                nn.ConvTranspose2d(in_channels=256, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                #1024xIxI
                nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2),
                #3x2Ix2I
            ))

class Discriminator(nn.Module):

    def __init__(self, input_dimension_size):
        super(Discriminator, self).__init__()

        self.level = 0
        self.input_dimension_size = input_dimension_size
        
        self.network = nn.Sequential()
        self.set_level(1)

        self.input = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.5))
        
        self.output = nn.Sequential(
                nn.Linear(128*self.input_dimension_size**2, 1),
                nn.Sigmoid())

    def forward(self, x):
        x = self.input(x)
        x = self.network(x).view(-1,128*self.input_dimension_size**2)
        return self.output(x)

    def set_level(self, level):
       self.level = level
       
       if "level_{}".format(level) not in self.network._modules.keys():
           self.add_level(level)     
        
        
    def add_level(self, level):
        
        # conv_output_size = (input_size - kernel_size + 2Padding)/stride + 1
        self.network.add_module("level_{}".format(level), nn.Sequential(
                #3xIxI
                nn.Conv2d(128, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.5),
                #512xIxI
                nn.Conv2d(1024, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(0.5)
                #128x(I/2)x(I/2)
            ))
        
        for l in range(level, 0, -1):
            self.network._modules.move_to_end("level_{}".format(l))
        
        
def main(): 
    input_dimension_size = 2
    g = Generator(input_dimension_size)
    d = Discriminator(input_dimension_size)
    
    print()
    print(g)
    print()
    print(d)
    print()
    
    print(g(torch.rand(25*input_dimension_size**2)).shape)
    print(d(g(torch.rand(25*input_dimension_size**2))))

    g.set_level(2)
    d.set_level(2)

    print()
    print(g)
    print()
    print(d)
    print()
    
    print(g(torch.rand(25*input_dimension_size**2)).shape)
    print(d(g(torch.rand(25*input_dimension_size**2))))
    
    g.set_level(3)
    d.set_level(3)

    print()
    print(g)
    print()
    print(d)
    print()
    
    print(g(torch.rand(25*input_dimension_size**2)).shape)
    print(d(g(torch.rand(25*input_dimension_size**2))))
    
    g.set_level(4)
    d.set_level(4)

    print()
    print(g)
    print()
    print(d)
    print()
    
    print(g(torch.rand(25*input_dimension_size**2)).shape)
    print(d(g(torch.rand(25*input_dimension_size**2))))
    
if __name__ == '__main__':
    main()
    
def init_weights_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if 'conv' in classname.lower() or 'linear' in classname.lower():
        nn.init.normal_(m.weight.data, mean=mean, std=std)   