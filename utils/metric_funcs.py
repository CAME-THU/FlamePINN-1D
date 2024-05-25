import numpy as np
from sklearn import metrics

# from . import config
from deepxde import config


def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))


def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def nanl2_relative_error(y_true, y_pred):
    """Return the L2 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err) / np.linalg.norm(y_true)


def l1_relative_error(y_true, y_pred):
    """equals to nMAPE"""
    return np.linalg.norm(y_true - y_pred, ord=1) / np.linalg.norm(y_true, ord=1)


def nanl1_relative_error(y_true, y_pred):
    """Return the L1 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err, ord=1) / np.linalg.norm(y_true, ord=1)


def linf_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, ord=np.inf) / np.linalg.norm(y_true, ord=np.inf)


def nanlinf_relative_error(y_true, y_pred):
    """Return the Linf relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err, ord=np.inf) / np.linalg.norm(y_true, ord=np.inf)


def mean_l2_relative_error(y_true, y_pred):
    """Compute the average of L2 relative error along the first axis."""
    return np.mean(
        np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
    )


def mean_squared_error(y_true, y_pred):
    """MSE"""
    return metrics.mean_squared_error(y_true, y_pred)


def root_mean_squared_error(y_true, y_pred):
    """RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """MAE"""
    return metrics.mean_absolute_error(y_true, y_pred)


def max_error(y_true, y_pred):
    """MaxE"""
    return metrics.max_error(y_true, y_pred)


def _absolute_percentage_error(y_true, y_pred):
    # return 100 * np.abs(
    #     (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(config.real(np)).eps, None)
    # )
    return np.abs(
        (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(config.real(np)).eps, None)
    )


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE"""
    return np.mean(_absolute_percentage_error(y_true, y_pred))


def max_absolute_percentage_error(y_true, y_pred):
    """maxAPE"""
    return np.amax(_absolute_percentage_error(y_true, y_pred))


def absolute_percentage_error_std(y_true, y_pred):
    """APE std"""
    return np.std(_absolute_percentage_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """R^2"""
    return metrics.r2_score(y_true, y_pred)


# for magnitude analysis: mean, min, max
def mean_absolute_true(y_true, y_pred):
    return np.mean(np.abs(y_true))


def mean_absolute_pred(y_true, y_pred):
    return np.mean(np.abs(y_pred))


def min_absolute_true(y_true, y_pred):
    return np.min(np.abs(y_true))


def min_absolute_pred(y_true, y_pred):
    return np.min(np.abs(y_pred))


def max_absolute_true(y_true, y_pred):
    return np.max(np.abs(y_true))


def max_absolute_pred(y_true, y_pred):
    return np.max(np.abs(y_pred))


def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "L2-RE": l2_relative_error,
        "nanl2 relative error": nanl2_relative_error,
        "nanL2-RE": nanl2_relative_error,
        "mean l2 relative error": mean_l2_relative_error,

        "l1 relative error": l1_relative_error,
        "L1-RE": l1_relative_error,
        "nanl1 relative error": nanl1_relative_error,
        "nanL1-RE": nanl1_relative_error,
        "linf relative error": linf_relative_error,
        "Linf-RE": linf_relative_error,
        "nanlinf relative error": nanlinf_relative_error,
        "nanLinf-RE": nanlinf_relative_error,

        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "RMSE": root_mean_squared_error,
        "rmse": root_mean_squared_error,
        "MAE": mean_absolute_error,
        "mae": mean_absolute_error,
        "MaxE": max_error,
        "maxE": max_error,

        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "MaxAPE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
        "APE std": absolute_percentage_error_std,

        "R2": r2_score,
        "r2": r2_score,

        "mean absolute of true": mean_absolute_true,
        "mean absolute of refe": mean_absolute_true,
        "mean absolute of pred": mean_absolute_pred,
        "min absolute of true": min_absolute_true,
        "min absolute of refe": min_absolute_true,
        "min absolute of pred": min_absolute_pred,
        "max absolute of true": max_absolute_true,
        "max absolute of refe": max_absolute_true,
        "max absolute of pred": max_absolute_pred,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret metric function identifier:", identifier)
