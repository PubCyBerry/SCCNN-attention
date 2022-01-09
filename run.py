import os
from omegaconf import OmegaConf

from src.runners import LOSO_Runner


def main(cfg=OmegaConf.load("config/config.yaml")) -> None:
    model_params = OmegaConf.load('config/models.yaml')
    cfg = OmegaConf.merge(cfg, model_params)
    cfg.merge_with_cli()

    runner = LOSO_Runner(
        log=cfg.log,
        optimizer=cfg.optimizer,
        loader=cfg.loader,
        network=cfg.network,
        data=cfg.data,
    )
    metrics = runner.run(profiler=cfg.get("profiler", "simple"))


if __name__ == "__main__":
    main()
