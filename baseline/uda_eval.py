"""!
@brief Visualize unsupervised EMG denoising results using a pre-trained SuDORM-RF model.
        Loads only mixture (no clean/noise), runs inference, and plots signals.

@author Your Name
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure local imports work
sys.path.append(os.getcwd())

from utils.config_loader import load_config
from models.improved_sudormrf import SuDORMRF
# Import the UNSUPERVISED dataloader function directly
from utils.emg_dataset import get_unsupervised_emg_dataloader


def load_model(checkpoint_path, hparams):
    model = SuDORMRF(
        out_channels=hparams.get('out_channels'),
        in_channels=hparams.get('in_channels'),
        num_blocks=hparams.get('num_blocks'),
        upsampling_depth=hparams.get('upsampling_depth'),
        enc_kernel_size=hparams.get('enc_kernel_size'),
        enc_num_basis=hparams.get('enc_num_basis'),
        num_sources=2,  # [clean_emg, noise]
    )
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded model from: {checkpoint_path}")
    return model


def apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams):
    if hparams.get("rescale_to_input_mixture", False):
        estimates = (estimates * mix_std) + mix_mean
    if hparams.get("apply_mixture_consistency", False):
        from utils.mixture_consistency import apply as apply_mixture_consistency
        estimates = apply_mixture_consistency(estimates, mixture)
    return estimates


def plot_signals(mixture, est_clean, est_noise, idx, save_dir):
    """
    Plot mixture, estimated clean, and estimated noise.
    All inputs: (1, L) tensors on CPU.
    """
    mixture = mixture.squeeze().cpu().numpy()
    est_clean = est_clean.squeeze().cpu().numpy()
    est_noise = est_noise.squeeze().cpu().numpy()
    L = len(mixture)
    t = range(L)

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, mixture, color='black', alpha=0.8)
    plt.title(f"Sample {idx}: Mixture (EMG + Noise)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, est_clean, color='green')
    plt.title("Estimated Clean EMG")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, est_noise, color='red')
    plt.title("Estimated Noise")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"sample_{idx:03d}.png"), dpi=150)
    plt.close()


def main():
    config_path = "config/eval_uda.yaml"
    hparams = load_config(config_path)

    # ✅ Use the correct unsupervised dataloader
    test_loader = get_unsupervised_emg_dataloader(
        data_dir=hparams["test_dir"],          # folder with single-channel .npy files (shape: (L,))
        batch_size=1,
        num_workers=hparams.get("num_workers", 4),
        fs=hparams.get("fs", 1000),
        segment_length=hparams.get("segment_length", 2000)
    )

    model = load_model(hparams["model_checkpoint"], hparams)
    model = model.cuda()
    model.eval()

    save_dir = hparams.get("plot_save_dir", "./plots_unsupervised")
    num_samples_to_plot = hparams.get("num_plot_samples", 10)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Plotting")):
            if i >= num_samples_to_plot:
                break

            mixture = batch["mixture"].cuda().unsqueeze(1)  # (1, 1, L)

            # Normalize (same as training)
            mix_std = mixture.std(dim=-1, keepdim=True)
            mix_mean = mixture.mean(dim=-1, keepdim=True)
            norm_mixture = (mixture - mix_mean) / (mix_std + 1e-9)

            # Inference
            estimates = model(norm_mixture)  # (1, 2, L)
            estimates = apply_output_transform(estimates, mix_std, mix_mean, mixture, hparams)

            est_clean = estimates[:, 0:1]   # (1, 1, L)
            est_noise = estimates[:, 1:2]   # (1, 1, L)

            # Plot and save
            plot_signals(mixture, est_clean, est_noise, i, save_dir)

    print(f"\n✅ Plotted {num_samples_to_plot} samples. Saved to: {save_dir}")


if __name__ == "__main__":
    main()