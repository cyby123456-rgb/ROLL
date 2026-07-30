import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args, overrides = parser.parse_known_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name, overrides=overrides)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config: RLVRConfig = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
