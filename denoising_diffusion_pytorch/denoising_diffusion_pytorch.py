import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn 
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torchvision import transforms as T
from tqdm.auto import tqdm

from loss.sceneflow_loss import sceneflow_loss_func

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta schedulers
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# diffusion model
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        timesteps = 20,
        sampling_timesteps = None,
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        #assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas) # beta_t
        register_buffer('alphas_cumprod', alphas_cumprod) # alpha_t_bar
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)) # 0
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod)) # 1
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod)) # 0
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod)) # inf
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # int
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # eq7 variance
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)  # 0
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20))) # -46.0517
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)) # eq7 coefficient a # 1
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)) # eq7 coefficient b # 0
        # offset noise strength - in blogpost, they claimed 0.1 was ideal
        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output[-1]
            #x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return  ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        #if clip_denoised:
        #    x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, self_cond, return_all_timesteps = False):
        batch, device = shape[0], self.device
        flow = torch.randn(shape, device = device)
        flows = [flow]
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = self_cond if self.self_condition else None
            flow, x_start = self.p_sample(flow, t, self_cond)
            flows.append(flow)
        ret = flow if not return_all_timesteps else torch.stack(flows, dim = 1)
        return ret


    @torch.inference_mode()
    def ddim_sample(self, shape, self_cond, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        flow = torch.randn(shape, device = device)
        flows = [flow]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = self_cond if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(flow, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)
            if time_next < 0:
                flow = x_start
                flows.append(flow)
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(flow).cuda()
            flow = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise # eq7
            flows.append(flow)
        ret = flow if not return_all_timesteps else torch.stack(flows, dim = 1)
        return ret

    @torch.inference_mode()
    def sample(self, pcs, return_all_timesteps = False):
        batch_size, _, num_points= pcs.shape
        device = pcs.device
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, 3, num_points), pcs, return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start)).cuda() # torch.Size([16, 8192, 3])
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + # B 3 N
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, pcs, t, noise = None, offset_noise_strength = None):
        b, _, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            #noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
            noise += offset_noise_strength * offset_noise[:,:,None,None]
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        x_self_cond = pcs
        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond) # torch.Size([16, 3, 8192]) torch.Size([16]) torch.Size([16, 8192, 6])
        if self.objective == 'pred_noise':
            target = noise.cuda()
        elif self.objective == 'pred_x0':
            target = x_start.cuda()
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v.cuda()
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss, metrics_3d = sceneflow_loss_func(model_out, target)
        #loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean(), metrics_3d

    def forward(self, flow_3d, pcs, *args, **kwargs):
        b, n, _, device, = *flow_3d.shape, flow_3d.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() # torch.Size([B])
        return self.p_losses(flow_3d, pcs, t, *args, **kwargs)
