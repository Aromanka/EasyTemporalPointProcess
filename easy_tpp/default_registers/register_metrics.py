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


def _get_masked_time_arrays(predictions, labels, **kwargs):
    """
    Extract flattened pred/label dtime arrays after applying seq_mask
    and optional EOS filtering.
    """
    seq_mask = kwargs.get("seq_mask", None)
    eos_token_id = kwargs.get("eos_token_id", None)

    pred = predictions[PredOutputIndex.TimePredIndex]
    label = labels[PredOutputIndex.TimePredIndex]

    if seq_mask is not None and len(seq_mask) > 0:
        pred = pred[seq_mask]
        label = label[seq_mask]

        if eos_token_id is not None:
            label_type = labels[PredOutputIndex.TypePredIndex][seq_mask]
        else:
            label_type = None
    else:
        label_type = labels[PredOutputIndex.TypePredIndex] if eos_token_id is not None else None

    pred = np.reshape(pred, [-1]).astype(np.float64)
    label = np.reshape(label, [-1]).astype(np.float64)

    valid = np.isfinite(pred) & np.isfinite(label)

    if eos_token_id is not None:
        label_type = np.reshape(label_type, [-1])
        valid = valid & (label_type != eos_token_id)

    pred = pred[valid]
    label = label[valid]

    normalize = kwargs.get("time_normalize", kwargs.get("normalize", "raw"))
    time_mean = kwargs.get("time_mean", 1.0)
    log_mean = kwargs.get("log_mean", None)
    log_std = kwargs.get("log_std", None)

    pred_raw = _to_raw_time(pred, normalize, time_mean, log_mean, log_std)
    label_raw = _to_raw_time(label, normalize, time_mean, log_mean, log_std)

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
