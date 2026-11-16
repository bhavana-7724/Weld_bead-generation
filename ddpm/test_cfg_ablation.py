import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler

NULL_CONDITION = torch.zeros(Config.COND_DIM)

@torch.no_grad()
def cfg_p_sample(model, scheduler, x, t, t_index, c, guidance_scale):
    b = x.shape[0]
    device = x.device
    
    # Evaluate conditional and unconditional
    c_null = NULL_CONDITION.unsqueeze(0).repeat(b, 1).to(device)
    uncond_pred = model(x, t, c_null)
    cond_pred = model(x, t, c)
    
    # CFG arithmetic
    noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    
    betas_t = scheduler._extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = scheduler._extract(scheduler.sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = scheduler._extract(scheduler.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def cfg_p_sample_loop(model, scheduler, shape, c, guidance_scale, seed=None):
    device = next(model.parameters()).device
    b = shape[0]
    
    if seed is not None:
        torch.manual_seed(seed)
    img = torch.randn(shape, device=device)
    
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc='CFG sampling', total=scheduler.timesteps, leave=False):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = cfg_p_sample(model, scheduler, img, t, i, c, guidance_scale)
        
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = (img * 255).type(torch.uint8)
    return img

def apply_condition_dropout(conds, dropout_prob=0.15):
    b = conds.shape[0]
    device = conds.device
    mask = torch.rand(b, device=device) < dropout_prob
    c_null = NULL_CONDITION.to(device).unsqueeze(0).repeat(b, 1)
    conds = torch.where(mask.unsqueeze(1), c_null, conds)
    return conds

def run_preflight():
    print("=== CFG PREFLIGHT VALIDATION ===")
    device = Config.DEVICE
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    
    # Test 1: Condition Dropout Logic
    print("[1] Testing condition dropout...")
    torch.manual_seed(42)
    fake_conds = torch.ones((1000, Config.COND_DIM))
    dropped_conds = apply_condition_dropout(fake_conds, dropout_prob=0.15)
    zeros_count = (dropped_conds.sum(dim=1) == 0).sum().item()
    print(f"    Target drop prob: 0.15. Actual drop fraction: {zeros_count / 1000.0:.3f}")
    assert 0.13 <= zeros_count / 1000.0 <= 0.17, "Dropout logic failed."
    
    # Test 2: CFG Sampling Equivalency (scale = 1.0)
    print("[2] Testing CFG sample equivalency at s=1.0...")
    model.eval()
    c = torch.ones((2, Config.COND_DIM), device=device)
    shape = (2, Config.IMG_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
    
    torch.manual_seed(42)
    orig_output = scheduler.p_sample_loop(model, shape, c)
    
    torch.manual_seed(42)
    cfg_output = cfg_p_sample_loop(model, scheduler, shape, c, guidance_scale=1.0)
    
    diff = torch.abs(orig_output.float() - cfg_output.float()).max().item()
    print(f"    Max pixel difference between original and CFG(s=1.0): {diff}")
    assert diff == 0, "CFG sampler at s=1.0 does not perfectly match original sampler."
    
    # Test 3: Gradient Updates
    print("[3] Testing gradient propagation...")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(shape, device=device)
    t = torch.tensor([50, 50], device=device).long()
    noise = torch.randn_like(x)
    x_t = scheduler.q_sample(x, t, noise)
    
    initial_cond_mlp_weight = model.cond_mlp[0].weight.clone()
    
    t0 = time.time()
    for _ in range(10): # 10 steps to measure time
        noise_pred = model(x_t, t, c)
        loss = F.mse_loss(noise_pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
    t1 = time.time()
    train_time_per_batch = (t1 - t0) / 10.0
    
    new_cond_mlp_weight = model.cond_mlp[0].weight.clone()
    weight_diff = torch.abs(new_cond_mlp_weight - initial_cond_mlp_weight).sum().item()
    print(f"    Weight diff after step: {weight_diff:.5f}")
    assert weight_diff > 0, "Weights did not update!"
    assert not torch.isnan(loss), "Loss is NaN."
    
    # Test 4: Checkpoint save/reload
    print("[4] Testing checkpointing...")
    torch.save(model.state_dict(), "test_cfg_chk.pt")
    model2 = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    model2.load_state_dict(torch.load("test_cfg_chk.pt"))
    os.remove("test_cfg_chk.pt")
    print("    Checkpoint save/load successful.")
    
    # Test 5: CFG Sampling Runtime
    print("[5] Measuring sampling runtime...")
    t0 = time.time()
    _ = cfg_p_sample_loop(model, scheduler, (1, 1, 256, 256), c[0:1], guidance_scale=3.0)
    t1 = time.time()
    sample_time = t1 - t0
    
    # Estimate total runtimes
    # Full dataset: 40 images. Batch size = 4 => 10 batches / epoch
    # 1000 epochs => 10,000 batches
    total_train_secs = 10000 * train_time_per_batch
    # Generation: 4 conditions * 5 seeds * 3 scales = 60 generations
    total_gen_secs = 60 * sample_time
    
    print("\n=== RUNTIME ESTIMATES ===")
    print(f"Time per training batch (bs=4): {train_time_per_batch:.3f} seconds")
    print(f"Estimated 1000-epoch training:  {total_train_secs/60.0:.1f} minutes")
    print(f"Time per CFG sample generation: {sample_time:.3f} seconds")
    print(f"Estimated 60-image generation:  {total_gen_secs/60.0:.1f} minutes")
    print(f"Total CPU execution estimate:   {(total_train_secs + total_gen_secs)/60.0:.1f} minutes")
    print("Preflight validation complete.")

if __name__ == "__main__":
    run_preflight()
