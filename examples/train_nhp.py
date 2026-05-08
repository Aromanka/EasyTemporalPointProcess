import argparse

import easy_tpp.default_registers  # noqa: F401
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
# pid=53928, gpu=2

python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id AttNHP_eval
nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id AttNHP_eval > logs/AttNHP_eval.log 2>&1 &
34421

nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id S2P2_train > logs/S2P2_train.log 2>&1 &
# pid=62380, gpu=0
python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id S2P2_eval

nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id NHP_train > logs/NHP_train.log 2>&1 &
# pid=37862, gpu=0
python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id NHP_eval

nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id SAHP_train > logs/SAHP_train.log 2>&1 &
# pid=18340, gpu=2
nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id SAHP_eval > logs/SAHP_eval.log 2>&1 &


nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id THP_train > logs/THP_train.log 2>&1 &
# pid=137444, gpu=0
python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id THP_eval


nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id RMTPP_train > logs/RMTPP_train.log 2>&1 &
# pid=34448, gpu=2
nohup python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id RMTPP_eval > logs/RMTPP_eval.log 2>&1 &

0504: RMTPP, NHP, S2P2, AttNHP


python examples/train_nhp.py --config_dir examples/configs/exp_config.yaml --experiment_id WSMTHP_train
"""
