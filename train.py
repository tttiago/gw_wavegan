import torch
import torch.optim as optim
from tqdm import tqdm

from models import WaveGANDiscriminator, WaveGANGenerator
from utils import get_noise, weights_init


def train_WaveGAN(train_loader, params):
    n_channels = params["n_channels"]
    waveform_length = params["waveform_length"]
    use_batchnorm = params["use_batchnorm"]
    cuda = params["cuda"]
    lr_g = params["lr_g"]
    lr_d = params["lr_d"]
    betas = params["betas"]
    n_epochs = params["n_epochs"]
    disc_repeats = params["disc_repeats"]
    z_dim = params["z_dim"]
    c_lambda = params["c_lambda"]
    display_step = params["display_lambda"]

    device = torch.device("cuda:0" if cuda else "cpu")

    generator_losses = []
    discriminator_losses = []

    # Set up and initialize generator.
    generator = WaveGANGenerator(
        n_channels=n_channels, output_size=waveform_length, use_batchnorm=use_batchnorm
    ).to(device)
    generator.apply(weights_init)

    # Set up and initialize discriminator.
    discriminator = WaveGANDiscriminator(
        n_channels=n_channels, input_size=waveform_length, use_batchnorm=use_batchnorm
    ).to(device)
    discriminator.apply(weights_init)

    # Set up Adam optimizers for both G and D.
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

    for epoch in tqdm(n_epochs):
        for real, _ in tqdm(train_loader):
            real = real.to(device)
            batch_size = len(real)
            epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

        mean_iteration_disc_loss = 0
        for _ in range(disc_repeats):
            # Update discriminator.
            optimizer_d.zero_grad()
            fake_noise = get_noise(batch_size, z_dim, device=device)
            fake = generator(fake_noise)
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(real)
            gradient = get_gradient(discriminator, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            disc_loss = calculate_disc_loss(disc_fake_pred, disc_real_pred, gp, c_lambda)

            # Keep track of the average discriminator loss in this batch.
            mean_iteration_disc_loss += disc_loss.item() / disc_repeats
            # Update gradients.
            disc_loss.backward(retain_graph=True)
            # Update weights.
            optimizer_d.step()
        discriminator_losses.append(mean_iteration_disc_loss)

        # Update generator.
        optimizer_g.zero_grad()
        fake_noise_2 = get_noise(batch_size, z_dim, device=device)
        fake_2 = generator(fake_noise_2)
        disc_fake_pred = discriminator(fake_2)

        gen_loss = calculate_gen_loss(disc_fake_pred)
        # Update gradients.
        gen_loss.backward()
        # Update weights.
        optimizer_g.step()

        generator_losses.append(gen_loss.item())

        print(
            f"Generator loss: {generator_losses[-1]} | Discriminator loss {discriminator_losses[-1]}"
        )


#     gan_model_name = "gan_{}.tar".format(model_prefix)

#     if take_backup and os.path.isfile(gan_model_name):
#         if cuda:
#             checkpoint = torch.load(gan_model_name)
#         else:
#             checkpoint = torch.load(gan_model_name, map_location="cpu")
#         self.generator.load_state_dict(checkpoint["generator"])
#         self.discriminator.load_state_dict(checkpoint["discriminator"])
#         self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
#         self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
#         self.train_d_cost = checkpoint["train_d_cost"]
#         self.train_w_distance = checkpoint["train_w_distance"]
#         self.valid_g_cost = checkpoint["valid_g_cost"]
#         self.g_cost = checkpoint["g_cost"]

#         first_iter = checkpoint["n_iterations"]
#         for i in range(0, first_iter, progress_bar_step_iter_size):
#             progress_bar.update()
#         self.generator.eval()
#         with torch.no_grad():
#             fake = self.generator(fixed_noise).detach().cpu().numpy()
#         save_samples(fake, first_iter)

#         if iter_indx % store_cost_every == 0:
#             self.g_cost.append(generator_cost.item() * -1)
#             self.train_d_cost.append(disc_cost.item())
#             self.train_w_distance.append(disc_wd.item() * -1)

#             progress_updates = {
#                 "Loss_D WD": str(self.train_w_distance[-1]),
#                 "Loss_G": str(self.g_cost[-1]),
#                 "Val_G": str(self.valid_g_cost[-1]),
#             }
#             progress_bar.set_postfix(progress_updates)

#         # lr decay
#         if decay_lr:
#             decay = max(0.0, 1.0 - (iter_indx * 1.0 / n_iterations))
#             # update the learning rate
#             update_optimizer_lr(self.optimizer_d, lr_d, decay)
#             update_optimizer_lr(self.optimizer_g, lr_g, decay)

#         if iter_indx % save_samples_every == 0:
#             with torch.no_grad():
#                 latent_space_interpolation(self.generator, n_samples=2)
#                 fake = self.generator(fixed_noise).detach().cpu().numpy()
#             save_samples(fake, iter_indx)

#         if take_backup and iter_indx % backup_every_n_iters == 0:
#             saving_dict = {
#                 "generator": self.generator.state_dict(),
#                 "discriminator": self.discriminator.state_dict(),
#                 "n_iterations": iter_indx,
#                 "optimizer_d": self.optimizer_d.state_dict(),
#                 "optimizer_g": self.optimizer_g.state_dict(),
#                 "train_d_cost": self.train_d_cost,
#                 "train_w_distance": self.train_w_distance,
#                 "valid_g_cost": self.valid_g_cost,
#                 "g_cost": self.g_cost,
#             }
#             torch.save(saving_dict, gan_model_name)


def get_gradient(disc, real, fake, epsilon):
    """
    Return the gradient of the discriminator's scores with respect to mixes of real and fake images.
    Parameters:
        disc: the discriminator model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the discriminator's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the discriminator's scores on the mixed images
    mixed_scores = disc(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the discriminator's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = (gradient_norm - 1).square().mean()
    assert not (torch.isnan(penalty))
    return penalty


def calculate_gen_loss(disc_fake_pred):
    """
    Return the loss of a generator given the discriminator's scores of the generator's fake images.
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    """
    gen_loss = -disc_fake_pred.mean()
    return gen_loss


def calculate_disc_loss(disc_fake_pred, disc_real_pred, gp, c_lambda):
    """
    Return the loss of a discriminator given the discriminator's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        disc_fake_pred: the discriminator's scores of the fake images
        disc_real_pred: the discriminator's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        disc_loss: a scalar for the discriminator's loss, accounting for the relevant factors
    """
    assert not (torch.isnan(disc_fake_pred.mean()))
    assert not (torch.isnan(disc_real_pred.mean()))

    disc_loss = disc_fake_pred.mean() - disc_real_pred.mean() + c_lambda * gp
    return disc_loss


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    manual_seed = 42
    torch.manual_seed(manual_seed)
    if cuda:
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.empty_cache()
