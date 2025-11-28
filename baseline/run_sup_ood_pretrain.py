"""!
@brief Training script for supervised SudoRM-RF on EMG denoising task.

@author Your Name
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pprint import pprint

# Ensure local imports work
sys.path.append(os.getcwd())

from utils.config_loader import load_config
from utils.cometml_logger import report_losses_mean_and_std
from utils.mixture_consistency import apply as apply_mixture_consistency
from models.improved_sudormrf import SuDORMRF
from utils.emg_logger import EMGLogger  
from utils.emg_dataset import get_emg_dataloaders
from losses.loss import negative_si_snr
import comet_ml


def apply_output_transform(rec_sources_wavs, input_mix_std,
                           input_mix_mean, input_mix, hparams):
    if hparams["rescale_to_input_mixture"]:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if hparams["apply_mixture_consistency"]:
        rec_sources_wavs = apply_mixture_consistency(rec_sources_wavs, input_mix)
    return rec_sources_wavs


def main():
    config_path = "config/emg_config.yaml"
    hparams = load_config(config_path)

    # Setup Comet.ml experiment (assuming already initialized globally as `experiment`)
    from comet_ml import Experiment
    global experiment
    experiment = Experiment(api_key=hparams["API_KEY"], project_name=hparams["project_name"])
    experiment.log_parameters(hparams)

    # Setup data
    train_loader, val_loader = get_emg_dataloaders(
        train_dir=hparams["train_dir"],
        val_dir=hparams["val_dir"],
        batch_size=hparams["batch_size"],
        num_workers=hparams.get("num_workers", 4)
    )

    # Initialize model
    model = SuDORMRF(
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources']  # should be 2: [clean, noise]
    )

    numparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    experiment.log_parameter('Parameters', numparams)
    print(f'Trainable Parameters: {numparams}')

    model = torch.nn.DataParallel(model).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

    # Initialize EMG logger
    emg_logger = EMGLogger(fs=hparams.get("fs", 1000.0), n_sources=2)

    # Paths
    checkpoint_storage_path = os.path.join(hparams["checkpoint_storage_path"], hparams["project_name"])
    if hparams["save_models_every"] > 0:
        os.makedirs(checkpoint_storage_path, exist_ok=True)

    # Training loop
    tr_step = 0
    val_step = 0

    for epoch in range(hparams['n_epochs']):
        print(f"Epoch {epoch + 1}/{hparams['n_epochs']}")

        # ------------------------
        # Training
        # ------------------------
        model.train()
        sum_loss = 0.0
        train_tqdm_gen = tqdm(train_loader, desc="Training")
        speaker_losses, noise_losses = [], []  

        for cnt, batch in enumerate(train_tqdm_gen):
            opt.zero_grad()

            clean = batch["clean"].cuda()   # (B, L)
            noise = batch["noise"].cuda()   # (B, L)

            gt_clean = clean.unsqueeze(1)   # (B, 1, L)
            mixture = (clean + noise).unsqueeze(1)  # (B, 1, L)

            # Normalize mixture
            mix_std = mixture.std(dim=-1, keepdim=True)
            mix_mean = mixture.mean(dim=-1, keepdim=True)
            norm_mixture = (mixture - mix_mean) / (mix_std + 1e-9)

            # Forward
            estimates = model(norm_mixture)  # (B, 2, L)
            estimates = apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams)

            est_clean = estimates[:, 0:1]   # (B, 1, L)
            est_noise = estimates[:, 1:2]   # (B, 1, L)

            # Loss
            clean_loss = torch.clamp(negative_si_snr(est_clean, gt_clean), min=-30., max=30.).mean()
            noise_loss = torch.clamp(negative_si_snr(est_noise, noise.unsqueeze(1)), min=-30., max=30.).mean()
            total_loss = 0.5 * clean_loss + 0.5 * noise_loss

            total_loss.backward()
            if hparams['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['clip_grad_norm'])
            opt.step()

            # Logging
            sum_loss += total_loss.item()
            speaker_losses.append(clean_loss.item())
            noise_losses.append(noise_loss.item())
            train_tqdm_gen.set_description(f"Train Loss: {sum_loss / (cnt + 1):.3f}")

        tr_step += 1

        # Learning rate scheduling
        if hparams['patience'] > 0 and tr_step % hparams['patience'] == 0:
            new_lr = hparams['learning_rate'] / (hparams['divide_lr_by'] ** (tr_step // hparams['patience']))
            print(f"Reducing LR to {new_lr}")
            for pg in opt.param_groups:
                pg['lr'] = new_lr

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_sisdr, val_sisdri = [], []
        with torch.no_grad():
            val_tqdm = tqdm(val_loader, desc="Validation")
            for batch in val_tqdm:
                clean = batch["clean"].cuda()
                noise = batch["noise"].cuda()

                gt_clean = clean.unsqueeze(1)
                mixture = (clean + noise).unsqueeze(1)

                mix_std = mixture.std(dim=-1, keepdim=True)
                mix_mean = mixture.mean(dim=-1, keepdim=True)
                norm_mixture = (mixture - mix_mean) / (mix_std + 1e-9)

                estimates = model(norm_mixture)
                estimates = apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams)

                est_clean = estimates[:, 0:1]

                sisdr_vals = -negative_si_snr(est_clean, gt_clean).cpu()
                mix_sisdr = -negative_si_snr(mixture, gt_clean).cpu()
                sisdri_vals = sisdr_vals - mix_sisdr

                val_sisdr.extend(sisdr_vals.tolist())
                val_sisdri.extend(sisdri_vals.tolist())

        # Log validation metrics
        res_dic = {
            "val_emg": {
                "sisdr": {"acc": val_sisdr},
                "sisdri": {"acc": val_sisdri}
            }
        }

        # Log EMG waveforms (only if enabled)
        if hparams.get("log_signals", False) and val_loader.dataset:
            # Take first batch for logging
            batch = next(iter(val_loader))
            clean = batch["clean"].cuda()
            noise = batch["noise"].cuda()
            gt_clean = clean.unsqueeze(1)
            mixture = (clean + noise).unsqueeze(1)

            mix_std = mixture.std(dim=-1, keepdim=True)
            mix_mean = mixture.mean(dim=-1, keepdim=True)
            norm_mixture = (mixture - mix_mean) / (mix_std + 1e-9)

            estimates = model(norm_mixture)
            estimates = apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams)

            emg_logger.log_emg_batch(
                experiment=experiment,
                est_clean=estimates[:, 0:1].detach(),
                est_noise=estimates[:, 1:2].detach(),
                gt_clean=gt_clean.detach(),
                gt_noise=noise.unsqueeze(1).detach(),
                mixture=mixture.detach(),
                step=tr_step,
                tag="val_emg",
                max_batch_items=4,
            )

        # Report metrics to Comet
        res_dic = report_losses_mean_and_std(res_dic, experiment, val_step)
        pprint(res_dic)
        val_step += 1

        # Save checkpoint
        if hparams["save_models_every"] > 0 and tr_step % hparams["save_models_every"] == 0:
            ckpt_path = os.path.join(checkpoint_storage_path, f"sup_teacher_epoch_{tr_step}.pt")
            torch.save(model.module.cpu().state_dict(), ckpt_path)
            model = model.cuda()  # move back to GPU

    experiment.end()


if __name__ == "__main__":
    main()