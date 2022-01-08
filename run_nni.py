import os
from omegaconf import OmegaConf
import nni

from src.runners import KFold_Runner


def transform_params(params):
    dot_list = [f"{key}={value}" for key, value in params.items()]
    return OmegaConf.from_dotlist(dot_list)


def main(conf=OmegaConf.load("config/config.yaml"), params=None) -> None:
    conf.merge_with_cli()
    assert params is not None
    params = transform_params(params)
    conf = OmegaConf.merge(conf, params)
    nni_path = os.environ["NNI_OUTPUT_DIR"]
    conf.log.log_path = os.path.join(nni_path, "tensorboard")
    conf.log.version = nni_path.split(os.path.sep)[-1]
    conf.log.checkpoint_path = os.path.join(conf.log.checkpoint_path, conf.log.version)

    runner = KFold_Runner(
        log=conf.log,
        optimizer=conf.optimizer,
        loader=conf.loader,
        network=conf.network,
        data=conf.data,
        setup=conf.inputs_setup,
        inputs=conf.get("setup", "s1"),
    )
    metrics = runner.run(profile=conf.get("profile", "simple"))
    nni.report_final_result(metrics["final_result"].item())


if __name__ == "__main__":
    params = nni.get_next_parameter()
    main(params=params)
