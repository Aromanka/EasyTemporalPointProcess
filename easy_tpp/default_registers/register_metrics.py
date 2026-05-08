import numpy as np

from easy_tpp.utils.const import PredOutputIndex
from easy_tpp.utils.metrics import MetricsHelper


_EPS = 1e-8


def _to_raw_time(x, normalize="raw", time_mean=1.0, log_mean=None, log_std=None):
    """
    Convert model-space inter-event time to raw inter-event time.

    normalize:
        raw/none: x is already raw time
        normal:  raw = clamp(x, 0) * time_mean
        log:     raw = exp(x * log_std + log_mean) * time_mean
    """
    x = np.asarray(x, dtype=np.float64)

    if normalize is None:
        normalize = "raw"
    normalize = str(normalize).lower()

    if normalize in ["raw", "none", "identity"]:
        return np.clip(x, a_min=0.0, a_max=None)

    if normalize == "normal":
        return np.clip(x, a_min=0.0, a_max=None) * float(time_mean)

    if normalize == "log":
        if log_mean is None or log_std is None:
            raise ValueError(
                "log_mean and log_std are required when normalize='log'."
            )
        return np.exp(x * float(log_std) + float(log_mean)) * float(time_mean)

    raise ValueError(f"Unsupported normalize mode: {normalize}")


def _first_nonzero_age_transition_mask(label_raw_2d, valid_mask_2d, eps=0.0):
    """
    找出每条序列中“累计 age 第一次变成非 0”的 transition。

    Args:
        label_raw_2d:
            shape = [batch_size, seq_len]
            已经还原到 raw scale 的 ground-truth inter-event time。

        valid_mask_2d:
            shape = [batch_size, seq_len]
            True 表示该位置参与评估；False 表示 padding、EOS 或非法值。

        eps:
            判断 age 是否非 0 的阈值。默认 0.0，即 age > 0 视为非 0。

    Returns:
        drop_mask:
            shape = [batch_size, seq_len]
            True 表示这个位置需要从 UKB time MAE/RMSE 中排除。
    """
    label_raw_2d = np.asarray(label_raw_2d, dtype=np.float64)
    valid_mask_2d = np.asarray(valid_mask_2d, dtype=bool)

    # padding / EOS / invalid 位置不应该影响累计 age
    safe_dtime = np.where(
        valid_mask_2d,
        np.clip(label_raw_2d, a_min=0.0, a_max=None),
        0.0,
    )

    # 用 label dtime 累加得到 age
    age = np.cumsum(safe_dtime, axis=1)

    # 每条序列中，age 第一次 > eps 的位置就是要排除的位置
    nonzero_age = (age > eps) & valid_mask_2d

    drop_mask = np.zeros_like(valid_mask_2d, dtype=bool)

    has_transition = np.any(nonzero_age, axis=1)
    if np.any(has_transition):
        rows = np.where(has_transition)[0]
        first_cols = np.argmax(nonzero_age[rows], axis=1)
        drop_mask[rows, first_cols] = True

    return drop_mask


def _get_masked_time_arrays(predictions, labels, **kwargs):
    """
    Extract flattened pred/label dtime arrays after applying seq_mask,
    optional EOS filtering, and optional UKB first-nonzero-age filtering.
    """
    seq_mask = kwargs.get("seq_mask", None)
    eos_token_id = kwargs.get("eos_token_id", None)

    # 新增开关：普通 mae/rmse 不启用；ukb_mae/ukb_rmse 启用
    exclude_first_nonzero_age_transition = kwargs.get(
        "exclude_first_nonzero_age_transition",
        False,
    )

    pred = np.asarray(
        predictions[PredOutputIndex.TimePredIndex],
        dtype=np.float64,
    )
    label = np.asarray(
        labels[PredOutputIndex.TimePredIndex],
        dtype=np.float64,
    )

    # 保持二维结构，因为“每条序列的第一个非 0 age transition”
    # 必须按 sequence 维度独立判断，不能先 flatten。
    if pred.ndim == 1:
        pred = pred[None, :]
    if label.ndim == 1:
        label = label[None, :]

    valid = np.ones(label.shape, dtype=bool)

    # 1. 应用 seq_mask，过滤 padding
    if seq_mask is not None and len(seq_mask) > 0:
        seq_mask = np.asarray(seq_mask, dtype=bool)
        if seq_mask.ndim == 1:
            seq_mask = seq_mask[None, :]
        valid = valid & seq_mask

    # 2. 应用 EOS 过滤
    if eos_token_id is not None:
        label_type = np.asarray(labels[PredOutputIndex.TypePredIndex])
        if label_type.ndim == 1:
            label_type = label_type[None, :]
        valid = valid & (label_type != eos_token_id)

    # 3. 读取时间反归一化参数
    normalize = kwargs.get("time_normalize", kwargs.get("normalize", "raw"))
    time_mean = kwargs.get("time_mean", 1.0)
    log_mean = kwargs.get("log_mean", None)
    log_std = kwargs.get("log_std", None)

    # 4. 先还原到 raw time
    pred_raw = _to_raw_time(pred, normalize, time_mean, log_mean, log_std)
    label_raw = _to_raw_time(label, normalize, time_mean, log_mean, log_std)

    # 5. 过滤 nan / inf
    valid = valid & np.isfinite(pred_raw) & np.isfinite(label_raw)

    # 6. UKB 口径：额外排除每条序列中第一个非 0 age transition
    if exclude_first_nonzero_age_transition:
        eps = kwargs.get("first_nonzero_age_eps", 0.0)
        drop_mask = _first_nonzero_age_transition_mask(
            label_raw,
            valid,
            eps=eps,
        )
        valid = valid & (~drop_mask)

    # 7. 最后再 flatten
    pred_raw = pred_raw[valid]
    label_raw = label_raw[valid]

    return pred_raw, label_raw


@MetricsHelper.register(name="rmse", direction=MetricsHelper.MINIMIZE, overwrite=True)
def rmse_metric_function(predictions, labels, **kwargs):
    pred_raw, label_raw = _get_masked_time_arrays(predictions, labels, **kwargs)
    if len(label_raw) == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred_raw - label_raw) ** 2)))


@MetricsHelper.register(name="mae", direction=MetricsHelper.MINIMIZE, overwrite=True)
def mae_metric_function(predictions, labels, **kwargs):
    pred_raw, label_raw = _get_masked_time_arrays(predictions, labels, **kwargs)
    if len(label_raw) == 0:
        return np.nan
    return float(np.mean(np.abs(pred_raw - label_raw)))


@MetricsHelper.register(name="ukb_rmse", direction=MetricsHelper.MINIMIZE, overwrite=True)
def ukb_rmse_metric_function(predictions, labels, **kwargs):
    """
    UKB-style RMSE for time prediction.

    Difference from normal rmse:
    each sequence's first transition where cumulative label age becomes non-zero
    is excluded from the time RMSE calculation.
    """
    pred_raw, label_raw = _get_masked_time_arrays(
        predictions,
        labels,
        exclude_first_nonzero_age_transition=True,
        **kwargs,
    )

    if len(label_raw) == 0:
        return np.nan

    return float(np.sqrt(np.mean((pred_raw - label_raw) ** 2)))


@MetricsHelper.register(name="ukb_mae", direction=MetricsHelper.MINIMIZE, overwrite=True)
def ukb_mae_metric_function(predictions, labels, **kwargs):
    """
    UKB-style MAE for time prediction.

    Difference from normal mae:
    each sequence's first transition where cumulative label age becomes non-zero
    is excluded from the time MAE calculation.
    """
    pred_raw, label_raw = _get_masked_time_arrays(
        predictions,
        labels,
        exclude_first_nonzero_age_transition=True,
        **kwargs,
    )

    if len(label_raw) == 0:
        return np.nan

    return float(np.mean(np.abs(pred_raw - label_raw)))


@MetricsHelper.register(name="rmse_log", direction=MetricsHelper.MINIMIZE, overwrite=True)
def rmse_log_metric_function(predictions, labels, **kwargs):
    pred_raw, label_raw = _get_masked_time_arrays(predictions, labels, **kwargs)
    if len(label_raw) == 0:
        return np.nan

    log_pred = np.log(np.clip(pred_raw, _EPS, None))
    log_label = np.log(np.clip(label_raw, _EPS, None))
    return float(np.sqrt(np.mean((log_pred - log_label) ** 2)))


@MetricsHelper.register(name="mae_log", direction=MetricsHelper.MINIMIZE, overwrite=True)
def mae_log_metric_function(predictions, labels, **kwargs):
    pred_raw, label_raw = _get_masked_time_arrays(predictions, labels, **kwargs)
    if len(label_raw) == 0:
        return np.nan

    log_pred = np.log(np.clip(pred_raw, _EPS, None))
    log_label = np.log(np.clip(label_raw, _EPS, None))
    return float(np.mean(np.abs(log_pred - log_label)))


@MetricsHelper.register(name="acc", direction=MetricsHelper.MAXIMIZE, overwrite=True)
def acc_metric_function(predictions, labels, **kwargs):
    seq_mask = kwargs.get("seq_mask")
    if seq_mask is None or len(seq_mask) == 0:
        pred = predictions[PredOutputIndex.TypePredIndex]
        label = labels[PredOutputIndex.TypePredIndex]
    else:
        pred = predictions[PredOutputIndex.TypePredIndex][seq_mask]
        label = labels[PredOutputIndex.TypePredIndex][seq_mask]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])

    eos_token_id = kwargs.get("eos_token_id", None)
    if eos_token_id is not None:
        valid = label != eos_token_id
        pred = pred[valid]
        label = label[valid]

    if len(label) == 0:
        return np.nan

    return float(np.mean(pred == label))


# @MetricsHelper.register(name='rmse', direction=MetricsHelper.MINIMIZE, overwrite=False)
# def rmse_metric_function(predictions, labels, **kwargs):
#     """Compute rmse metrics of the time predictions.

#     Args:
#         predictions (np.array): model predictions.
#         labels (np.array): ground truth.

#     Returns:
#         float: average rmse of the time predictions.
#     """
#     seq_mask = kwargs.get('seq_mask')
#     if seq_mask is None or len(seq_mask) == 0:
#         # If mask is empty or None, use all predictions
#         pred = predictions[PredOutputIndex.TimePredIndex]
#         label = labels[PredOutputIndex.TimePredIndex]
#     else:
#         pred = predictions[PredOutputIndex.TimePredIndex][seq_mask]
#         label = labels[PredOutputIndex.TimePredIndex][seq_mask]

#     pred = np.reshape(pred, [-1])
#     label = np.reshape(label, [-1])
#     return np.sqrt(np.mean((pred - label) ** 2))


@MetricsHelper.register(name='acc', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def acc_metric_function(predictions, labels, **kwargs):
    """Compute accuracy ratio metrics of the type predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: accuracy ratio of the type predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    if seq_mask is None or len(seq_mask) == 0:
        # If mask is empty or None, use all predictions
        pred = predictions[PredOutputIndex.TypePredIndex]
        label = labels[PredOutputIndex.TypePredIndex]
    else:
        pred = predictions[PredOutputIndex.TypePredIndex][seq_mask]
        label = labels[PredOutputIndex.TypePredIndex][seq_mask]
    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.mean(pred == label)

@MetricsHelper.register(name="acc_gt_time", direction=MetricsHelper.MAXIMIZE, overwrite=True)
def acc_gt_time_metric_function(predictions, labels, **kwargs):
    """
    Accuracy of next type prediction conditioned on ground-truth next event time.
    predictions[2] should be pred_type_gt_time.
    """
    if len(predictions) < 3 or predictions[2] is None:
        raise ValueError(
            "acc_gt_time requires predictions[2]. "
            "Please return (pred_dtime, pred_type, pred_type_gt_time) in TorchModelWrapper.run_batch()."
        )

    seq_mask = kwargs.get("seq_mask")
    if seq_mask is None or len(seq_mask) == 0:
        pred = predictions[2]
        label = labels[1]
    else:
        pred = predictions[2][seq_mask]
        label = labels[1][seq_mask]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])

    eos_token_id = kwargs.get("eos_token_id", None)
    if eos_token_id is not None:
        valid = label != eos_token_id
        pred = pred[valid]
        label = label[valid]

    if len(label) == 0:
        return np.nan

    return float(np.mean(pred == label))
