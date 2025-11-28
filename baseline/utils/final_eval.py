"""!
@brief Final evaluation of a pre-trained SuDORM-RF model on EMG denoising task.
        Configuration is loaded from a YAML file (e.g., eval.yaml).

@author Your Name
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pprint import pprint
import pickle

# Ensure local imports work
sys.path.append(os.getcwd())

from utils.config_loader import load_config
from utils.mixture_consistency import apply as apply_mixture_consistency
from models.improved_sudormrf import SuDORMRF
from utils.emg_dataset import get_emg_dataloaders
from losses.loss import negative_si_snr


def correct_sign(est, ref):
    """
    Flip the sign of `est` if it improves correlation with `ref`.
    Inputs: (B, 1, L) or (B, L)
    Returns: sign-corrected `est` with same shape.
    """
    if est.ndim == 3:
        est = est.squeeze(1)  # (B, L)
    if ref.ndim == 3:
        ref = ref.squeeze(1)  # (B, L)

    B, L = est.shape
    est_np = est.cpu().numpy()
    ref_np = ref.cpu().numpy()
    corrected = np.zeros_like(est_np)

    for b in range(B):
        e = est_np[b]
        r = ref_np[b]

        # Avoid division by zero / constant signals
        if np.std(e) < 1e-8 or np.std(r) < 1e-8:
            corr = 0.0
        else:
            corr = np.corrcoef(e, r)[0, 1]

        # Flip sign if correlation is negative
        if corr < 0:
            corrected[b] = -e
        else:
            corrected[b] = e

    return torch.from_numpy(corrected).to(est.device).unsqueeze(1)  # (B, 1, L)


def compute_pearson_corr(est, ref):
    """
    Compute Pearson correlation coefficient per batch sample.
    Inputs: (B, 1, L) or (B, L)
    Returns: numpy array of shape (B,)
    """
    if est.ndim == 3:
        est = est.squeeze(1)
    if ref.ndim == 3:
        ref = ref.squeeze(1)
    
    est = est.cpu().numpy()
    ref = ref.cpu().numpy()
    corr = []
    for e, r in zip(est, ref):
        if np.std(e) < 1e-8 or np.std(r) < 1e-8:
            corr.append(0.0)
        else:
            c = np.corrcoef(e, r)[0, 1]
            corr.append(float(c) if not np.isnan(c) else 0.0)
    return np.array(corr)


def load_model(checkpoint_path, hparams):
    model = SuDORMRF(
        out_channels=hparams.get('out_channels', 256),
        in_channels=hparams.get('in_channels', 512),
        num_blocks=hparams.get('num_blocks', 8),
        upsampling_depth=hparams.get('upsampling_depth', 7),
        enc_kernel_size=hparams.get('enc_kernel_size', 81),
        enc_num_basis=hparams.get('enc_num_basis', 512),
        num_sources=2,  # [clean_emg, noise]
    )
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Handle DataParallel checkpoints
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded model from: {checkpoint_path}")
    return model


def apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams):
    if hparams.get("rescale_to_input_mixture", False):
        estimates = (estimates * mix_std) + mix_mean
    if hparams.get("apply_mixture_consistency", False):
        estimates = apply_mixture_consistency(estimates, mixture)
    return estimates


def _agg(arr):
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr))
    }


def main():
    # Load hyperparameters from YAML
    config_path = "config/eval.yaml"
    hparams = load_config(config_path)

    # Setup test dataloader
    _, test_loader = get_emg_dataloaders(
        train_dir=None,
        val_dir=hparams["test_dir"],
        batch_size=hparams.get("batch_size", 1),
        num_workers=hparams.get("num_workers", 4)
    )

    # Load model
    model = load_model(hparams["model_checkpoint"], hparams)
    model = model.cuda()
    model.eval()

    all_sisdr_raw = []
    all_sisdri_raw = []
    all_corr_raw = []

    all_sisdr_corrected = []
    all_sisdri_corrected = []
    all_corr_corrected = []

    test_tqdm_gen = tqdm(enumerate(test_loader), desc="EMG Denoising Eval", total=len(test_loader))

    with torch.no_grad():
        for j, batch in test_tqdm_gen:
            clean = batch["clean"].cuda()      # (B, L)
            noise = batch["noise"].cuda()      # (B, L)

            gt_clean = clean.unsqueeze(1)      # (B, 1, L)
            mixture = (clean + noise).unsqueeze(1)  # (B, 1, L)

            # Normalize mixture (same as training)
            mix_std = mixture.std(dim=-1, keepdim=True)
            mix_mean = mixture.mean(dim=-1, keepdim=True)
            norm_mixture = (mixture - mix_mean) / (mix_std + 1e-9)

            # Forward pass
            estimates = model(norm_mixture)    # (B, 2, L)
            estimates = apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams)

            est_clean_raw = estimates[:, 0:1]  # (B, 1, L)

            # --- Raw metrics ---
            sisdr_raw = -negative_si_snr(est_clean_raw, gt_clean).cpu().numpy()
            mix_sisdr = -negative_si_snr(mixture, gt_clean).cpu().numpy()
            sisdri_raw = sisdr_raw - mix_sisdr
            corr_raw = compute_pearson_corr(est_clean_raw, gt_clean)

            all_sisdr_raw.extend(sisdr_raw.tolist())
            all_sisdri_raw.extend(sisdri_raw.tolist())
            all_corr_raw.extend(corr_raw.tolist())

            # --- Sign-corrected metrics ---
            est_clean_corrected = correct_sign(est_clean_raw, gt_clean)
            sisdr_corrected = -negative_si_snr(est_clean_corrected, gt_clean).cpu().numpy()
            sisdri_corrected = sisdr_corrected - mix_sisdr
            corr_corrected = compute_pearson_corr(est_clean_corrected, gt_clean)

            all_sisdr_corrected.extend(sisdr_corrected.tolist())
            all_sisdri_corrected.extend(sisdri_corrected.tolist())
            all_corr_corrected.extend(corr_corrected.tolist())

            # Update progress bar
            avg_sisdri = np.mean(all_sisdri_corrected)
            test_tqdm_gen.set_description(
                f"SI-SDRi (sign-corrected): {avg_sisdri:.2f} ({j+1}/{len(test_loader)})"
            )

    # Aggregate results
    aggregate_results = {
        "raw": {
            "sisdr": _agg(all_sisdr_raw),
            "sisdri": _agg(all_sisdri_raw),
            "pearson_corr": _agg(all_corr_raw)
        },
        "corrected": {
            "sisdr": _agg(all_sisdr_corrected),
            "sisdri": _agg(all_sisdri_corrected),
            "pearson_corr": _agg(all_corr_corrected)
        }
    }

    pprint(aggregate_results)

    # Save results
    model_name = os.path.splitext(os.path.basename(hparams["model_checkpoint"]))[0]
    save_dir = hparams.get("save_results_dir", "/tmp")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_emg_eval_results.pkl")

    with open(save_path, 'wb') as f:
        pickle.dump(aggregate_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ Evaluation complete. Results saved to:\n{save_path}")


if __name__ == "__main__":
    main()