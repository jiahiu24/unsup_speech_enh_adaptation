import torch

def si_snr(x, s, eps=1e-8):
    """
    Calculate SI-SNR for each sample in the batch.
    Args:
        x: [B, T] or [B, 1, T]
        s: [B, T] or [B, 1, T]
    Returns:
        si_snr_vals: [B], in dB
    """
    # Remove extra dimensions
    if x.dim() == 3:
        x = x.squeeze(1)
    if s.dim() == 3:
        s = s.squeeze(1)

    # Zero-mean
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)

    # Optimal scaling factor
    alpha = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) / (torch.sum(s_zm ** 2, dim=-1, keepdim=True) + eps)

    # Target and noise components
    s_target = alpha * s_zm
    e_noise = x_zm - s_target

    # SI-SNR
    si_snr_vals = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return si_snr_vals  # [B]

def negative_si_snr(x, s, eps=1e-8):
    return -si_snr(x, s, eps)