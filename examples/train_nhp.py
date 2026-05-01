import argparse

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)

    model_runner = Runner.build_from_config(config)

    model_runner.run()


if __name__ == '__main__':
    main()

"""
# AttNHP
python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id AttNHP_train
nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id AttNHP_train > logs/AttNHP_train.log 2>&1 &
# pid=67671
python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id AttNHP_eval
nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id AttNHP_eval > logs/AttNHP_eval.log 2>&1 &
34421

python examples/train_nhp.py --config_dir examples/configs/ai4s_exp_config.yaml --experiment_id S2P2_train
python examples/train_nhp.py --config_dir examples/configs/ai4s_exp_config.yaml --experiment_id S2P2_eval
"""
