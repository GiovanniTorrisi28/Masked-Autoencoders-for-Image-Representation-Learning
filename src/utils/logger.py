from pathlib import Path

import torch


class UnifiedLogger:
    """
    Logs metrics simultaneously to TensorBoard and Weights & Biases.

    Args:
        log_dir: directory where TensorBoard writes event files
        experiment_name: used as wandb run name and sub-folder of log_dir
        config: full experiment config dict, passed to wandb.init
        use_wandb: enable W&B logging (default True)
        use_tensorboard: enable TensorBoard logging (default True)
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str,
        config: dict,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        tb_dir = Path(log_dir) / experiment_name

        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(tb_dir))
        else:
            self._writer = None

        if self.use_wandb:
            import wandb
            wandb_project = config.get("logging", {}).get("wandb_project", "mae-project")
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=config,
                dir=str(tb_dir),
                resume="allow",
            )
            self._wandb = wandb
        else:
            self._wandb = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar to TensorBoard and/or W&B."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def log_scalars(self, tag_value_dict: dict[str, float], step: int) -> None:
        """Log multiple scalars in one call."""
        for tag, value in tag_value_dict.items():
            if self._writer is not None:
                self._writer.add_scalar(tag, value, global_step=step)
        if self._wandb is not None:
            self._wandb.log(tag_value_dict, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """
        Log an image tensor (C, H, W) or (B, C, H, W).
        Pixel values should be in [0, 1].
        """
        if self._writer is not None:
            if image.dim() == 3:
                self._writer.add_image(tag, image, global_step=step)
            else:
                self._writer.add_images(tag, image, global_step=step)
        if self._wandb is not None:
            import wandb
            if image.dim() == 3:
                img = image.permute(1, 2, 0).cpu().numpy()
                self._wandb.log({tag: wandb.Image(img)}, step=step)
            else:
                imgs = [wandb.Image(img.permute(1, 2, 0).cpu().numpy()) for img in image]
                self._wandb.log({tag: imgs}, step=step)

    def finish(self) -> None:
        """Close TensorBoard writer and finish W&B run."""
        if self._writer is not None:
            self._writer.close()
        if self._wandb is not None:
            self._wandb.finish()
