import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


#############################
# Model Utils
#############################
def weights_init(m):
    """Initialize weights of model."""
    if isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


# def set_gradients_status(model, flag):
#     for p in model.parameters():
#         p.requires_grad = flag


# def update_optimizer_lr(optimizer, lr, decay):
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr * decay


#############################
# Sampling noise for model.
#############################
def get_noise(n_samples, z_dim, device="cpu"):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


#############################
# File Utils
#############################

# def make_path(output_path):
#     if not os.path.isdir(output_path):
#         os.makedirs(output_path)
#     return output_path


#############################
# Plotting utils
#############################


def visualize_loss(loss_1, loss_2, first_legend, second_legend, y_label):
    plt.figure(figsize=(10, 5))
    plt.title("{} and {} Loss During Training".format(first_legend, second_legend))
    plt.plot(loss_1, label=first_legend)
    plt.plot(loss_2, label=second_legend)
    plt.xlabel("iterations")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    if not (os.path.isdir("visualization")):
        os.makedirs("visualization")
    plt.savefig("visualization/loss.png")
