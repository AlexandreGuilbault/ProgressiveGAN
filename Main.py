import torch
from Train import Coach

# Some great ideas taken from here : https://github.com/soumith/ganhacks

#####################
# Settings
num_train_images = 10000
batch_size = 32
input_dimension_size = 2
n_epochs = 50

# Two Timescale Update Rule (TTUR) : https://medium.com/beyondminds/advances-in-generative-adversarial-networks-7bad57028032
# "Typically, a slower update rule is used for the generator and a faster update rule is used for the discriminator"
learning_rate_discriminator = 0.0008
learning_rate_generator = 0.0002

beta_l = 0.5
beta_h = 0.999

data_dir = 'processed-celeba-small'
#####################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

coach = Coach(input_dimension_size=input_dimension_size, num_train_images=num_train_images, data_dir=data_dir, device=device)

##################################
# Train Level 1
coach.set_level(level=1, d_lr=learning_rate_discriminator, g_lr=learning_rate_generator, batch_size=batch_size, beta_l=beta_l, beta_h=beta_h)
coach.train(n_epochs=n_epochs)
coach.save()
# coach.load(1)
# coach.generate_random_image(level=1)

##################################
# Train Level 2

coach.set_level(level=2, d_lr=learning_rate_discriminator/2, g_lr=learning_rate_generator/2, batch_size=batch_size//2, beta_l=beta_l, beta_h=beta_h)
coach.train(n_epochs=n_epochs*2)
coach.save()
# coach.load(2)
# coach.generate_random_image(level=2)

##################################
# Train Level 3

coach.set_level(level=3, d_lr=learning_rate_discriminator/4, g_lr=learning_rate_generator/4, batch_size=batch_size//4, beta_l=beta_l, beta_h=beta_h)
coach.train(n_epochs=n_epochs*4)
coach.save()
# coach.load(3)
# coach.generate_random_image(level=3)

##################################
# Train Level 4

coach.set_level(level=4, d_lr=learning_rate_discriminator/8, g_lr=learning_rate_generator/8, batch_size=batch_size//8, beta_l=beta_l, beta_h=beta_h)
coach.train(n_epochs=n_epochs*8)
coach.save()
# coach.load(4)
# coach.generate_random_image(level=4)

##################################
# Train Level 5

coach.set_level(level=5, d_lr=learning_rate_discriminator/16, g_lr=learning_rate_generator/16, batch_size=batch_size//16, beta_l=beta_l, beta_h=beta_h)
coach.train(n_epochs=n_epochs*16)
coach.save()
# coach.load(5)
# coach.generate_random_image(level=5)

##################################
# Train Level 6

coach.set_level(level=6, d_lr=learning_rate_discriminator/32, g_lr=learning_rate_generator/32, batch_size=batch_size//32, beta_l=beta_l, beta_h=beta_h)
coach.train(n_epochs=n_epochs*32)
coach.save()
# coach.load(6)
# coach.generate_random_image(level=6)


##################################
# Print images

coach.print_results_per_epoch(level=2, step=5)

coach.generate_random_image(level=2)
coach.print_random_image()
