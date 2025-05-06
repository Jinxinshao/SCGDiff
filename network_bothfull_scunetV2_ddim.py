import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from math import pi
from .loss import Gradient_Loss, ColorHistogramLoss
from .loss import L1_Charbonnier_loss as Charbonnier_Loss
from PIL import Image
import torch.nn.functional as F


class Network(BaseNetwork):  # SCUnet_V2
    def __init__(self, unet, beta_schedule, module_name='scunet_convnext', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        elif module_name == 'scunet_convnext':
            from .scunet.scunet_convnext import SCUNet

        self.denoise_fn = SCUNet(**unet)
        self.beta_schedule = beta_schedule
        self.w_G_Ang = torch.tensor(0.5, requires_grad=True)
        self.loss_l1_charbonnier_SR = torch.tensor(0.5, requires_grad=True)
        self.m_G_Ang = torch.tensor(0.0)
        self.loss_l1_charbonnier_SR = torch.tensor(0.0)
        self.learning_param = torch.nn.Parameter(torch.tensor(0.1))
        self.initial_values = torch.tensor([0.6, 0.7, 0.8, 0.9])

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def ssim(img1, img2, window_size=11, k1=0.01, k2=0.03, C1=1e-4, C2=1e-3):
        N, C, H, W = img1.size()
        window = torch.ones((C, 1, window_size, window_size)).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C) / (window_size ** 2)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C) / (window_size ** 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) / (window_size ** 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) / (window_size ** 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) / (window_size ** 2) - mu1_mu2

        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def p_sample(self, x, t, t_index, y_cond=None):
        b = x.shape[0]
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(x.device)
        e_t = self.denoise_fn(torch.cat([y_cond, x], dim=1), noise_level)

        alpha_t = extract(self.alphas_cumprod, t, x.shape)
        alpha_t_prev = extract(self.alphas_cumprod, (t - 1).clamp(min=0), x.shape)

        sigma_t = 0.01  # Small non-zero value for more stability
        pred_x0 = (x - (1 - alpha_t).sqrt() * e_t) / alpha_t.sqrt()
        dir_xt = (1 - alpha_t_prev - sigma_t**2).sqrt() * e_t
        x_prev = alpha_t_prev.sqrt() * pred_x0 + dir_xt + sigma_t * torch.randn_like(x)

        return x_prev

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=50):
        b, *_ = y_cond.shape

        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        time_steps = torch.linspace(0, self.num_timesteps - 1, sample_num).long().to(y_cond.device)
        time_steps = time_steps.flip(0)

        for i, step in enumerate(time_steps):
            index = self.num_timesteps - step - 1
            t = torch.full((b,), step, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, index, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t

        return y_t, y_t

    def forward(self, y_0, y_cond=None, mask=None, noise=None, ssim=ssim):
        Gradient_loss = Gradient_Loss().cuda()
        Charbonnier_loss = Charbonnier_Loss().cuda()
        compute_color_histogram_loss = ColorHistogramLoss().cuda()
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        mask = self.update_mask(mask, self.learning_param, self.initial_values)

        if mask is not None:
            mean_all = torch.mean(mask)
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*(mask + 1 - mean_all) + (mean_all - mask)*y_0], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
            loss_G_Ang = compute_color_histogram_loss(noise, noise_hat)
            loss_l1_charbonnier_SR = Charbonnier_loss(noise, noise_hat)
            loss_gradient_SR = Gradient_loss(noise, noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
            loss_G_Ang = compute_color_histogram_loss(noise, noise_hat)
            loss_l1_charbonnier_SR = Charbonnier_loss(noise, noise_hat)
            loss_gradient_SR = Gradient_loss(noise, noise_hat)


        combined_loss = 0.5*loss + 0.5*loss_G_Ang +  99.0*loss_gradient_SR
        return combined_loss

# Gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# Beta schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
