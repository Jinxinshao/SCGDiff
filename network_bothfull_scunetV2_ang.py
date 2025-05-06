import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from math import pi
from .loss import Gradient_Loss,ColorHistogramLoss
from .loss import L1_Charbonnier_loss as Charbonnier_Loss
#from torchvision import rgb_to_hsv
#import torch
from PIL import Image
import torch.nn.functional as F


class Network(BaseNetwork):#SCUnet_V2
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
        self.initial_values = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])

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




        
    def argular_loss(self, illum_gt, illum_pred):
        # img_gt = img_input / illum_gt
        # illum_gt = img_input / img_gt
        # illum_pred = img_input / img_output
   
        # ACOS
        cos_between = torch.nn.CosineSimilarity(dim=1)
        cos = cos_between(illum_gt, illum_pred)
        cos = torch.clamp(cos,-0.99999, 0.99999)
        loss = torch.mean(torch.acos(cos)) * 180 / pi

   		# MSE
        # loss = torch.mean((illum_gt - illum_pred)**2)
   
        # 1 - COS
        # loss = 1 - torch.mean(cos)
   
        # 1 - COS^2
        # loss = 1 - torch.mean(cos**2)
        return loss
      
    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn
    def softmax(self, w1, w2):
        return torch.exp(w1) / (torch.exp(w1) + torch.exp(w2)), torch.exp(w2) / (torch.exp(w1) + torch.exp(w2))
    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))#

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )
    def update_mask(self, mask, learning_param , initial_values , tolerance=0.01):
        for i in range(len(initial_values)):
        # 计算新值
            new_value = 0.6 + i * learning_param
        # 更新 mask 中接近 initial_values[i] 的值
            mask = torch.where(
                torch.abs(mask - initial_values[i]) < tolerance,
                new_value,
                mask
            )
        return mask
    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None,ssim =ssim):
        # sampling from p(gammas)
        Gradient_loss = Gradient_Loss().cuda()
        Charbonnier_loss = Charbonnier_Loss().cuda()
        #color_histogram_loss
        compute_color_histogram_loss = ColorHistogramLoss().cuda()
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        mask = self.update_mask(mask, self.learning_param, self.initial_values)
        
        if mask is not None:
            mean_all = torch.mean(mask)
            #print(mean_all)
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*(mask + 1 -mean_all)+(mean_all - mask)*y_0], dim=1), sample_gammas)#
            loss = self.loss_fn(noise, noise_hat)
            loss_G_Ang = compute_color_histogram_loss(noise, noise_hat)
            loss_l1_charbonnier_SR = Charbonnier_loss(noise, noise_hat)          
            loss_gradient_SR = Gradient_loss(noise, noise_hat)
            #loss_L_lab = L_lab(noise, noise_hat)
            #loss_L_lch = L_lch(noise, noise_hat)
            #loss_ssim = 1 - ssim(noise, noise_hat)
            #loss = loss
            #loss =loss + loss_G_Ang
            
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)#
            loss = self.loss_fn(noise, noise_hat)
            loss_G_Ang = compute_color_histogram_loss(noise, noise_hat)
            loss_l1_charbonnier_SR = Charbonnier_loss(noise, noise_hat)           
            loss_gradient_SR = Gradient_loss(noise, noise_hat)
            #loss_L_lab = L_lab(noise, noise_hat)
            #loss_L_lch = L_lch(noise, noise_hat)
            #print(noise.shape,noise_hat.shape)
            #loss_ssim = 1 - ssim(noise, noise_hat)
            #loss = loss
            #loss = loss + loss_G_Ang
      

        # 使用新的权重组合loss
        print('0.001*loss_G_Ang', 0.001*loss_G_Ang)
        print('1*loss',1*loss)
        print('0.999*loss_gradient_SR',0.999*loss_gradient_SR)
        combined_loss = 0.001*loss_G_Ang  + 0.999*loss_gradient_SR #+ 1*loss
        return combined_loss


# gaussian diffusion trainer class
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

# beta_schedule function
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


