"""
@brief Training script for RemixIT-based unsupervised adaptation on EMG denoising.

@author Your Name
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pprint import pprint

sys.path.append(os.getcwd())

from utils.config_loader import load_config
from utils.cometml_logger import report_losses_mean_and_std  # 可选，若不用可删
from utils.mixture_consistency import apply as apply_mixture_consistency
from models.improved_sudormrf import SuDORMRF
from utils.emg_logger import EMGLogger  
from utils.emg_dataset import get_unsupervised_emg_dataloader
from losses.loss import negative_si_snr
import comet_ml


def apply_output_transform(rec_sources_wavs, input_mix_std,
                           input_mix_mean, input_mix, hparams):
    if hparams["rescale_to_input_mixture"]:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if hparams["apply_mixture_consistency"]:
        rec_sources_wavs = apply_mixture_consistency(rec_sources_wavs, input_mix)
    return rec_sources_wavs


def remix_signals(est_clean, est_noise):
    device = est_clean.device
    B = est_clean.shape[0]

    swap = (torch.rand(B, device=device) < 0.5)
    swap = swap.unsqueeze(-1).unsqueeze(-1)

    remixed_clean = torch.where(swap, est_noise, est_clean)
    remixed_noise = torch.where(swap, est_clean, est_noise)

    scale_clean = 1.0 + 0.1 * torch.randn_like(remixed_clean, device=device)
    scale_noise = 1.0 + 0.1 * torch.randn_like(remixed_noise, device=device)

    remixed_clean = remixed_clean * scale_clean
    remixed_noise = remixed_noise * scale_noise

    return remixed_clean, remixed_noise


def update_teacher(student_model, teacher_model, momentum=0.99):
    with torch.no_grad():
        for param_q, param_k in zip(student_model.parameters(), teacher_model.parameters()):
            param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data


def main():
    config_path = "config/emg_remixit_config.yaml"
    hparams = load_config(config_path)

    from comet_ml import Experiment
    global experiment
    experiment = Experiment(api_key=hparams["API_KEY"], project_name=hparams["project_name"])
    experiment.log_parameters(hparams)

    # ----------------------------
    # Data Loader: Unsupervised Only
    # ----------------------------
    unsup_train_loader = get_unsupervised_emg_dataloader(
        data_dir=hparams["unsup_train_dir"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"]
        # 不再传 fs
    )

    # ----------------------------
    # Models: Student & Teacher
    # ----------------------------
    def build_model():
        return SuDORMRF(
            out_channels=hparams['out_channels'],
            in_channels=hparams['in_channels'],
            num_blocks=hparams['num_blocks'],
            upsampling_depth=hparams['upsampling_depth'],
            enc_kernel_size=hparams['enc_kernel_size'],
            enc_num_basis=hparams['enc_num_basis'],
            num_sources=2  # clean + noise
        )

    student_model = build_model()
    teacher_model = build_model()

    # Load warmup checkpoint if provided
    if hparams.get("warmup_checkpoint"):
        ckpt = torch.load(hparams["warmup_checkpoint"], map_location="cpu")
        student_model.load_state_dict(ckpt)
        teacher_model.load_state_dict(ckpt)
        print(f"Loaded warmup checkpoint from {hparams['warmup_checkpoint']}")

    student_model = torch.nn.DataParallel(student_model).cuda()
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    teacher_model.eval()

    numparams = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    experiment.log_parameter('Parameters', numparams)
    print(f'Trainable Parameters: {numparams}')

    opt = torch.optim.Adam(student_model.parameters(), lr=hparams['learning_rate'])

    emg_logger = EMGLogger(fs=hparams["fs"], n_sources=2)

    checkpoint_storage_path = os.path.join(hparams["checkpoint_storage_path"], hparams["project_name"])
    if hparams["save_models_every"] > 0:
        os.makedirs(checkpoint_storage_path, exist_ok=True)

    # ----------------------------
    # Training Loop
    # ----------------------------
    tr_step = 0

    for epoch in range(hparams['n_epochs']):
        print(f"Epoch {epoch + 1}/{hparams['n_epochs']}")

        student_model.train()
        sum_loss = 0.0
        train_tqdm_gen = tqdm(unsup_train_loader, desc="RemixIT Train")

        for cnt, batch in enumerate(train_tqdm_gen):
            opt.zero_grad()

            mixture = batch["mixture"].cuda().unsqueeze(1)  # [B, 1, T]

            # Normalize
            mix_std = mixture.std(dim=-1, keepdim=True)
            mix_mean = mixture.mean(dim=-1, keepdim=True)
            norm_mixture = (mixture - mix_mean) / (mix_std + 1e-9)

            # Teacher inference
            with torch.no_grad():
                teacher_estimates = teacher_model(norm_mixture)  # [B, 2, T]
                teacher_estimates = apply_output_transform(
                    teacher_estimates, mix_std, mix_mean, mixture, hparams
                )
                est_clean_t = teacher_estimates[:, 0:1]
                est_noise_t = teacher_estimates[:, 1:2]

            # RemixIT
            remixed_clean, remixed_noise = remix_signals(est_clean_t, est_noise_t)
            new_mixture = remixed_clean + remixed_noise

            # Normalize new mixture
            new_mix_std = new_mixture.std(dim=-1, keepdim=True)
            new_mix_mean = new_mixture.mean(dim=-1, keepdim=True)
            norm_new_mixture = (new_mixture - new_mix_mean) / (new_mix_std + 1e-9)

            # Student forward
            student_estimates = student_model(norm_new_mixture)
            student_estimates = apply_output_transform(
                student_estimates, new_mix_std, new_mix_mean, new_mixture, hparams
            )
            est_clean_s = student_estimates[:, 0:1]
            est_noise_s = student_estimates[:, 1:2]

            # Loss
            clean_loss = torch.clamp(negative_si_snr(est_clean_s, remixed_clean), min=-30., max=30.).mean()
            noise_loss = torch.clamp(negative_si_snr(est_noise_s, remixed_noise), min=-30., max=30.).mean()
            total_loss = 0.5 * clean_loss + 0.5 * noise_loss

            total_loss.backward()a
            if hparams['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), hparams['clip_grad_norm'])
            opt.step()

            # EMA update teacher
            if hparams.get("teacher_momentum", 0.99) > 0:
                update_teacher(student_model.module, teacher_model.module, hparams["teacher_momentum"])

            sum_loss += total_loss.item()
            train_tqdm_gen.set_description(f"Train Loss: {sum_loss / (cnt + 1):.3f}")

        tr_step += 1

        # Learning rate scheduling
        if hparams['patience'] > 0 and tr_step % hparams['patience'] == 0:
            new_lr = hparams['learning_rate'] / (hparams['divide_lr_by'] ** (tr_step // hparams['patience']))
            for pg in opt.param_groups:
                pg['lr'] = new_lr

        # Optional: Log example signals from training batch
        if hparams.get("log_signals", False) and (epoch + 1) % hparams.get("log_every_epoch", 1) == 0:
            with torch.no_grad():
                # Reuse last batch for logging
                estimates_vis = student_model(norm_mixture)
                estimates_vis = apply_output_transform(estimates_vis, mix_std, mix_mean, mixture, hparams)

                emg_logger.log_emg_batch(
                    experiment=experiment,
                    est_clean=estimates_vis[:, 0:1].detach(),
                    est_noise=estimates_vis[:, 1:2].detach(),
                    gt_clean=None,
                    gt_noise=None,
                    mixture=mixture.detach(),
                    step=tr_step,
                    tag="train_remixit",
                    max_batch_items=min(4, len(mixture))
                )

        # Save model periodically
        if hparams["save_models_every"] > 0 and (epoch + 1) % hparams["save_models_every"] == 0:
            ckpt_path = os.path.join(checkpoint_storage_path, f"remixit_student_epoch_{epoch+1}.pt")
            torch.save(student_model.module.state_dict(), ckpt_path)

    experiment.end()


if __name__ == "__main__":
    main()