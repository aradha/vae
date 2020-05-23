import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

NUM_CLASSES = 10

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.num_latent_vars = 128
        self.conv_encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(16, 16, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(16, 32, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 32, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 64, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 64, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 128, 3, padding=1),
                                          nn.ReLU())

        self.mean_encoder = nn.Sequential(nn.Linear(128 * 32 * 32 + NUM_CLASSES,
                                                    self.num_latent_vars))
        self.covariance_encoder = nn.Sequential(nn.Linear(128 * 32 * 32 + NUM_CLASSES, self.num_latent_vars))

        self.linear_decoder = nn.Sequential(nn.Linear(self.num_latent_vars + NUM_CLASSES, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 32 * 32 * 3))
        self.conv_decoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(16, 16, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(16, 32, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 32, 3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 3, 3, padding=1),
                                          nn.Sigmoid())

        #for m in self.modules():
            #if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #init.kaiming_normal(m.weight.data)
                #m.bias.data.fill_(0.1)


    def encode(self, images, labels):
        convs = self.conv_encoder(images)
        convs = convs.view(-1, 128 * 32 * 32)
        mean = self.mean_encoder(torch.cat((convs, labels), 1))
        # Assume diagonal covariance matrix
        log_covariance = self.covariance_encoder(torch.cat((convs, labels), 1))
        return mean, log_covariance

    # Gaussian prior/posteriors
    def sample_latent_vars(self, mean, log_covariance):
        eps = torch.cuda.FloatTensor(mean.size()).normal_()
        eps = Variable(eps)
        latent_vars = mean + (log_covariance.mul(.5).exp()).mul(eps)
        return latent_vars

    def decode(self, latent_vars, labels):
        linear = self.linear_decoder(torch.cat((latent_vars, labels), 1))
        conv = linear.view(-1, 3, 32, 32)
        return self.conv_decoder(conv)

    def generate(self, num_samples, labels):
        shape = torch.Size([num_samples, self.num_latent_vars])
        latent_vars = torch.cuda.FloatTensor(shape).normal_()
        latent_vars = Variable(latent_vars)
        decoded = self.decode(latent_vars, labels)
        return decoded

    def forward(self, inputs, labels):
        mean, log_covar = self.encode(inputs, labels)
        latent_vars = self.sample_latent_vars(mean, log_covar)
        decoded = self.decode(latent_vars, labels)
        return decoded, mean, log_covar
