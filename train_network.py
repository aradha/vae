import model as m
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
import visdom
import time
import random

NUM_CLASSES = 10

def train_network(train_loader, test_loader):
    model = m.VAE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 10000
    vis = visdom.Visdom('http://192.168.0.14')

    best_test_loss = float("inf")
    for i in range(num_epochs):
        print("Epoch: ", (i + 1))
        start = time.time()
        train_loss = train_step(train_loader, model, optimizer)
        end = time.time()
        print("Train Time: ", end - start)
        print("Train Loss: ", train_loss)
        #test_loss = test_step(test_loader, model)
        test_loss = 0.
        print("Test Loss: ", test_loss)
        if test_loss <= best_test_loss:
            print("SAVING NEW MODEL")
            best_test_loss = test_loss
            save_model(model)
            test_samples, reconstructions = get_examples(test_loader, model)
            visualize_examples(vis, test_samples, reconstructions,
                               'test_samples')
            train_samples, reconstructions = get_examples(train_loader, model)
            visualize_examples(vis, train_samples, reconstructions,
                               'train_samples')
            num_samples = 5
            sample_labels = []
            for i in range(num_samples):
                sample_labels.append(random.randint(0,9))
            t_sample_labels = densify_labels(sample_labels)
            t_sample_labels = Variable(t_sample_labels).cuda()
            samples = model.generate(num_samples, t_sample_labels)
            visualize_samples(vis, samples, sample_labels)
        print("Best Loss: ", best_test_loss)

def loss(outputs, targets, mean, log_covariance):

    num_latents = mean.size()[1]

    kld = mean.view(-1, 1, num_latents).bmm(mean.view(-1, num_latents, 1))
    kld -= num_latents
    #print(kld.size())
    #print(torch.sum(log_covariance.exp(), 1).size())
    #print(torch.sum(log_covariance, 1).size())
    kld += torch.sum(log_covariance.exp(), 1).view(-1, 1, 1)
    kld -= torch.sum(log_covariance, 1).view(-1, 1, 1)
    kld = .5 * kld
    kld = kld.sum()

    bce = nn.MSELoss(size_average=False)
    bce_loss = bce(outputs, targets)
    #print(outputs[0,:])
    #print(bce_loss)
    return bce_loss + kld

def densify_labels(labels):
    dense_labels = np.zeros((len(labels), NUM_CLASSES))
    for idx, l in enumerate(labels):
        dense_labels[idx, l] = 1.
    return torch.from_numpy(dense_labels).float()

def train_step(train_loader, model, optimizer):
    train_loss = 0.0
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = Variable(inputs).cuda()
        labels = densify_labels(labels)
        labels = Variable(labels).cuda()

        outputs, mean, log_covariance = model(inputs, labels)
        batch_loss = loss(outputs, inputs, mean, log_covariance)
        batch_loss.backward()
        train_loss += batch_loss.data[0]
        optimizer.step()
        break
    return train_loss / len(train_loader.dataset)

def test_step(test_loader, model):
    test_loss = 0.0
    model.eval()
    for idx, batch in enumerate(test_loader):
        inputs, labels = batch
        inputs = Variable(inputs, volatile=True).cuda()
        labels = densify_labels(labels)
        labels = Variable(labels, volatile=True).cuda()
        outputs, mean, covariance = model(inputs, labels)
        batch_loss = loss(outputs, inputs, mean, covariance)
        batch_loss.backward()
        test_loss += batch_loss.data[0]
    model.train()
    return test_loss / len(test_loader.dataset)

def save_model(model):
    torch.save(model, 'cifar10-vae.pt')

def visualize_samples(vis, samples, labels):
    vis.close(env='samples')
    for idx, s in enumerate(samples):
        vis.image(s.cpu().data, env='samples',
                  opts=dict(title='Sample: ' + str(labels[idx])))

def get_examples(loader, model, num_samples=5):

    for i, (data, labels) in enumerate(loader):
        test_examples = data[0:num_samples, :]
        labels = labels[0:num_samples]
        labels = densify_labels(labels)
        labels = Variable(labels).cuda()
        test_examples = Variable(test_examples).cuda()
        reconstructions, _, _ = model(test_examples, labels)
        break

    return test_examples, reconstructions

def visualize_examples(vis, test_samples, reconstructions,
                      env_name):
    vis.close(env=env_name)
    for idx, s in enumerate(test_samples):
        original = s.cpu().data.numpy()
        reconstruction = reconstructions[idx].cpu().data.numpy()
        stack = np.stack([original, reconstruction])
        vis.images(stack, env=env_name, nrow=2,
                   opts=dict(title='Sample: ' + str(idx + 1)))
