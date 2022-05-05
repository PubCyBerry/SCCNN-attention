from omegaconf import OmegaConf
from pathlib import Path

from src import runners

CONFIG_DIR = Path("Configs")


def main(cfg=OmegaConf.load(CONFIG_DIR / "config.yaml")) -> None:
    model_params = OmegaConf.load(CONFIG_DIR / "models.yaml")
    cfg = OmegaConf.merge(cfg, model_params)
    cfg.merge_with_cli()

    runner = getattr(runners, cfg.runner)(
        log=cfg.log,
        optimizer=cfg.optimizer,
        loader=cfg.loader,
        network=cfg.network,
        data=cfg.data,
    )
    metrics = runner.run(profiler=cfg.get("profiler", "simple"))


if __name__ == "__main__":
    main()
