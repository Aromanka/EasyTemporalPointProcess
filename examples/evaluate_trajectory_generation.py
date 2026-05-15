import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _event_set_jaccard(pred_types, gt_types):
    pred_set = set(pred_types.tolist())
    gt_set = set(gt_types.tolist())

    if len(gt_set) == 0:
        return np.nan

    union = pred_set | gt_set
    if len(union) == 0:
        return 0.0

    return len(pred_set & gt_set) / len(union)


def _event_multiset_f1(pred_types, gt_types):
    if len(gt_types) == 0:
        return np.nan

    pred_counter = Counter(pred_types.tolist())
    gt_counter = Counter(gt_types.tolist())

    overlap = sum((pred_counter & gt_counter).values())

    precision = overlap / max(len(pred_types), 1)
    recall = overlap / max(len(gt_types), 1)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _position_type_acc(pred_types, gt_types):
    k = min(len(pred_types), len(gt_types))
    if k == 0:
        return np.nan
    return float(np.mean(pred_types[:k] == gt_types[:k]))


def _position_time_mae(pred_times, gt_times):
    k = min(len(pred_times), len(gt_times))
    if k == 0:
        return np.nan
    return float(np.mean(np.abs(pred_times[:k] - gt_times[:k])))


def _filter_future(types, times, cutoff_time, max_time, min_event_id):
    types = np.asarray(types)
    times = np.asarray(times)

    mask = (
        (times > cutoff_time)
        & (times <= max_time)
        & (types >= min_event_id)
    )
    return types[mask], times[mask]


def build_eval_samples(
    data_loader,
    samples: Optional[int],
    cutoff_time: Optional[float],
    history_ratio: float,
    max_time: Optional[float],
    min_history_len: int,
) -> List[Dict]:
    """
    Build EasyTPP trajectory-eval samples.

    EasyTPP batch fields:
      time_seqs: absolute time since sequence start
      time_delta_seqs: inter-event time
      type_seqs: event type id
      seq_non_pad_mask: valid event mask
    """
    eval_samples = []

    for batch in data_loader:
        time_seqs, dtime_seqs, type_seqs, seq_non_pad_mask, _ = batch.values()

        time_seqs = _to_numpy(time_seqs)
        dtime_seqs = _to_numpy(dtime_seqs)
        type_seqs = _to_numpy(type_seqs)
        seq_non_pad_mask = _to_numpy(seq_non_pad_mask).astype(bool)

        batch_size = type_seqs.shape[0]

        # print(f"Evaluation batch_size={batch_size}")

        for i in range(batch_size):
            valid = seq_non_pad_mask[i]

            times = time_seqs[i][valid].astype(np.float32)
            dtimes = dtime_seqs[i][valid].astype(np.float32)
            types = type_seqs[i][valid].astype(np.int64)

            if len(times) < min_history_len + 1:
                continue

            if cutoff_time is not None:
                this_cutoff = float(cutoff_time)
                hist_mask = times <= this_cutoff
                if hist_mask.sum() < min_history_len:
                    continue
            else:
                cut_idx = max(min_history_len, int(len(times) * history_ratio))
                cut_idx = min(cut_idx, len(times) - 1)
                hist_mask = np.zeros_like(times, dtype=bool)
                hist_mask[:cut_idx] = True
                this_cutoff = float(times[cut_idx - 1])

            this_max_time = float(max_time) if max_time is not None else float(times[-1])
            future_mask = (times > this_cutoff) & (times <= this_max_time)

            if hist_mask.sum() < min_history_len:
                continue
            if future_mask.sum() == 0:
                continue

            sample = {
                "input_times": times[hist_mask],
                "input_dtimes": dtimes[hist_mask],
                "input_types": types[hist_mask],
                "gt_times": times[future_mask],
                "gt_types": types[future_mask],
                "input_len": int(hist_mask.sum()),
                "cutoff_time": this_cutoff,
                "max_time": this_max_time,
            }
            eval_samples.append(sample)

            if samples is not None and len(eval_samples) >= samples:
                return eval_samples

    if len(eval_samples) == 0:
        raise ValueError(
            "No valid evaluation samples found. "
            "Try lowering --min_history_len, changing --cutoff_time, "
            "or using --history_ratio."
        )

    return eval_samples


@torch.no_grad()
def rollout_one_sample(
    model,
    sample: Dict,
    device,
    max_new_events: int,
    sample_dtime: bool,
    sample_type: bool,
    temperature: float,
):
    time_seq = torch.tensor(
        sample["input_times"][None, :],
        dtype=torch.float32,
        device=device,
    )
    dtime_seq = torch.tensor(
        sample["input_dtimes"][None, :],
        dtype=torch.float32,
        device=device,
    )
    type_seq = torch.tensor(
        sample["input_types"][None, :],
        dtype=torch.long,
        device=device,
    )

    out = model.generate_trajectory(
        time_seq=time_seq,
        time_delta_seq=dtime_seq,
        event_seq=type_seq,
        max_steps=max_new_events,
        max_time=sample["max_time"],
        include_prefix=True,
        sample_dtime=sample_dtime,
        sample_type=sample_type,
        temperature=temperature,
        return_dict=True,
    )

    full_times = _to_numpy(out["time_seq"][0]).astype(np.float32)
    full_types = _to_numpy(out["event_seq"][0]).astype(np.int64)
    full_mask = _to_numpy(out["seq_non_pad_mask"][0]).astype(bool)

    input_len = sample["input_len"]

    pred_times = full_times[input_len:]
    pred_types = full_types[input_len:]
    pred_mask = full_mask[input_len:]

    pred_times = pred_times[pred_mask]
    pred_types = pred_types[pred_mask]

    return pred_types, pred_times, full_types, full_times, full_mask


def evaluate_samples(
    model,
    eval_samples: List[Dict],
    device,
    max_new_events: int,
    min_event_id: int,
    sample_dtime: bool,
    sample_type: bool,
    temperature: float,
    log_jsonl_path: Optional[str],
):
    model.eval()

    rows = []

    log_fp = None
    if log_jsonl_path is not None:
        os.makedirs(os.path.dirname(log_jsonl_path), exist_ok=True)
        log_fp = open(log_jsonl_path, "w", encoding="utf-8")

    for sample_idx, sample in enumerate(tqdm(eval_samples, desc="trajectory-eval")):
        pred_types, pred_times, full_types, full_times, full_mask = rollout_one_sample(
            model=model,
            sample=sample,
            device=device,
            max_new_events=max_new_events,
            sample_dtime=sample_dtime,
            sample_type=sample_type,
            temperature=temperature,
        )

        pred_types_f, pred_times_f = _filter_future(
            pred_types,
            pred_times,
            cutoff_time=sample["cutoff_time"],
            max_time=sample["max_time"],
            min_event_id=min_event_id,
        )
        gt_types_f, gt_times_f = _filter_future(
            sample["gt_types"],
            sample["gt_times"],
            cutoff_time=sample["cutoff_time"],
            max_time=sample["max_time"],
            min_event_id=min_event_id,
        )

        metrics = {
            "sample_idx": sample_idx,
            "event_jaccard": _event_set_jaccard(pred_types_f, gt_types_f),
            "event_multiset_f1": _event_multiset_f1(pred_types_f, gt_types_f),
            "position_type_acc": _position_type_acc(pred_types_f, gt_types_f),
            "position_time_mae": _position_time_mae(pred_times_f, gt_times_f),
            "num_pred": int(len(pred_types_f)),
            "num_gt": int(len(gt_types_f)),
            "length_error": int(len(pred_types_f) - len(gt_types_f)),
            "cutoff_time": float(sample["cutoff_time"]),
            "max_time": float(sample["max_time"]),
        }

        rows.append(metrics)

        if log_fp is not None:
            log_fp.write(json.dumps({
                "sample_idx": sample_idx,
                "input_types": sample["input_types"].tolist(),
                "input_times": sample["input_times"].tolist(),
                "pred_types": pred_types_f.tolist(),
                "pred_times": pred_times_f.tolist(),
                "gt_types": gt_types_f.tolist(),
                "gt_times": gt_times_f.tolist(),
                "metrics": metrics,
            }, ensure_ascii=False) + "\n")

    if log_fp is not None:
        log_fp.close()

    return rows


def summarize(rows: List[Dict]):
    keys = [
        "event_jaccard",
        "event_multiset_f1",
        "position_type_acc",
        "position_time_mae",
        "num_pred",
        "num_gt",
        "length_error",
    ]

    summary = {}
    for key in keys:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(np.nanmean(vals))

    summary["num_samples"] = len(rows)
    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, use model_config.pretrained_model_dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "valid", "val", "test"],
    )
    parser.add_argument("--samples", type=int, default=None)

    parser.add_argument(
        "--cutoff_time",
        type=float,
        default=None,
        help="Absolute cutoff time. If omitted, use --history_ratio per sequence.",
    )
    parser.add_argument(
        "--history_ratio",
        type=float,
        default=0.5,
        help="Used only when --cutoff_time is omitted.",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=None,
        help="Global max generation/eval time. If omitted, use each sequence's last GT time.",
    )
    parser.add_argument("--min_history_len", type=int, default=2)
    parser.add_argument("--max_new_events", type=int, default=50)

    parser.add_argument(
        "--min_event_id",
        type=int,
        default=0,
        help="Ignore event ids smaller than this value in metrics.",
    )

    parser.add_argument("--sample_dtime", action="store_true")
    parser.add_argument("--sample_type", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--output_dir", type=str, default="trajectory_eval_outputs")

    args = parser.parse_args()

    config = Config.build_from_yaml_file(
        args.config_dir,
        experiment_id=args.experiment_id,
    )

    runner = Runner.build_from_config(config)

    if args.ckpt is not None:
        runner._load_model(args.ckpt)

    model = runner.model
    device = model.device
    print(f"model on device: {device}")

    split = {"valid": "dev", "val": "dev"}.get(args.split, args.split)
    data_loader = runner._data_loader.get_loader(split=split, shuffle=False)

    eval_samples = build_eval_samples(
        data_loader=data_loader,
        samples=args.samples,
        cutoff_time=args.cutoff_time,
        history_ratio=args.history_ratio,
        max_time=args.max_time,
        min_history_len=args.min_history_len,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_jsonl_path = os.path.join(args.output_dir, "trajectory_logs.jsonl")

    rows = evaluate_samples(
        model=model,
        eval_samples=eval_samples,
        device=device,
        max_new_events=args.max_new_events,
        min_event_id=args.min_event_id,
        sample_dtime=args.sample_dtime,
        sample_type=args.sample_type,
        temperature=args.temperature,
        log_jsonl_path=log_jsonl_path,
    )

    summary = summarize(rows)

    metrics_path = os.path.join(args.output_dir, "trajectory_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_sample": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Trajectory generation evaluation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved trajectory logs to: {log_jsonl_path}")


if __name__ == "__main__":
    main()

"""
# run commands
python examples/evaluate_trajectory_generation.py \
  --config_dir examples/configs/exp_config.yaml \
  --experiment_id S2P2_eval \
  --split val \
  --cutoff_time 21915.0 \
  --max_time 31067.5 \
  --samples 100 \
  --max_new_events 100 \
  --output_dir outputs/nhp_ukb_eval

# run commands for ukb_cd_norm
nohup \
python examples/evaluate_trajectory_generation.py \
  --config_dir examples/configs/exp_config.yaml \
  --experiment_id S2P2_eval \
  --split val \
  --cutoff_time 15.92428 \
  --max_time 22.5748 \
  --max_new_events 100 \
  --samples 4000 \
  --output_dir outputs/s2p2_ukb_eval \
> logs/S2P2_eval_gen.log 2>&1 &
pid = 1487838, gpu=0
"""