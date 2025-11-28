# baseline/utils/emg_logger.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class EMGLogger:
    def __init__(
        self,
        fs: float = 1000.0,          # EMG 采样率 (Hz)
        n_sources: int = 2,          # 预期输出源数量（如 clean + noise）
        max_channels: int = 8,       # 最大同时显示通道数（防图太密）
    ):
        self.fs = fs
        self.n_sources = n_sources
        self.max_channels = max_channels

    def log_emg_batch(
        self,
        experiment,
        est_clean: "torch.Tensor",   # (B, C, L) or (B, 1, L)
        est_noise: Optional["torch.Tensor"] = None,  # 可选
        gt_clean: Optional["torch.Tensor"] = None,
        gt_noise: Optional["torch.Tensor"] = None,
        mixture: Optional["torch.Tensor"] = None,
        step: int = 0,
        tag: str = "val",
        max_batch_items: int = 4,
    ):
        """
        Logs EMG signal plots to Comet.ml.

        Args:
            experiment: comet_ml.Experiment instance
            est_clean: estimated clean EMG, shape (B, C, L)
            est_noise: estimated noise (optional)
            gt_clean: ground truth clean EMG (optional)
            gt_noise: ground truth noise (optional)
            mixture: input mixture (optional)
            step: training step for logging
            tag: dataset name (e.g., 'val_emg')
            max_batch_items: max number of samples to log
        """
        B = min(est_clean.shape[0], max_batch_items)
        C = est_clean.shape[1]
        L = est_clean.shape[2]

        # 时间轴
        t = np.arange(L) / self.fs  # seconds

        for b in range(B):
            fig = self._plot_emg_sample(
                t=t,
                est_clean=est_clean[b].cpu().numpy(),
                est_noise=est_noise[b].cpu().numpy() if est_noise is not None else None,
                gt_clean=gt_clean[b].cpu().numpy() if gt_clean is not None else None,
                gt_noise=gt_noise[b].cpu().numpy() if gt_noise is not None else None,
                mixture=mixture[b].cpu().numpy() if mixture is not None else None,
                C=min(C, self.max_channels),
            )

            experiment.log_figure(
                figure_name=f"{tag}_emg_sample_{b}_step_{step}",
                figure=fig,
                step=step,
            )
            plt.close(fig)

    def _plot_emg_sample(self, t, est_clean, est_noise=None, gt_clean=None,
                        gt_noise=None, mixture=None, C=1):
        """内部绘图函数"""
        n_rows = 1  # 至少有 est_clean
        has_gt = gt_clean is not None
        has_mixture = mixture is not None
        has_est_noise = est_noise is not None
        has_gt_noise = gt_noise is not None

        # 确定子图数量
        rows = []
        if has_mixture:
            rows.append("mixture")
        if has_gt:
            rows.append("gt_clean")
        rows.append("est_clean")
        if has_est_noise:
            rows.append("est_noise")
        if has_gt_noise:
            rows.append("gt_noise")

        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        row_idx = 0

        # Mixture
        if has_mixture:
            self._plot_channels(axes[row_idx], t, mixture, title="Mixture", color='gray')
            row_idx += 1

        # Ground Truth Clean
        if has_gt:
            self._plot_channels(axes[row_idx], t, gt_clean, title="Ground Truth EMG", color='green')
            row_idx += 1

        # Estimated Clean
        self._plot_channels(axes[row_idx], t, est_clean, title="Estimated EMG", color='blue')
        row_idx += 1

        # Estimated Noise
        if has_est_noise:
            self._plot_channels(axes[row_idx], t, est_noise, title="Estimated Noise", color='red')
            row_idx += 1

        # Ground Truth Noise
        if has_gt_noise:
            self._plot_channels(axes[row_idx], t, gt_noise, title="Ground Truth Noise", color='orange')

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        return fig

    def _plot_channels(self, ax, t, signal, title, color):
        """绘制多通道信号（最多 self.max_channels）"""
        C = min(signal.shape[0], self.max_channels)
        for c in range(C):
            label = f"Ch {c}" if C > 1 else None
            ax.plot(t, signal[c], color=color, alpha=0.8, label=label)
        ax.set_ylabel(title)
        if C > 1:
            ax.legend(loc='upper right', fontsize='small')