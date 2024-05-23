from typing import Tuple

import torch
from torch import Tensor
import math

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.regression import MeanSquaredError

'''
The following module contrains a couple of metrics that work as R^2 replacements. 
The main issue with R^2 is that when the image has very few class, 
for example the image is mostly water then the mean is quite low, leading to negative R^2 scores. 
'''


def _r2_score_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute R2 score.

    Checks for same shape and 1D/2D input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    _check_same_shape(preds, target)
    if preds.ndim > 2:
        raise ValueError(
            "Expected both prediction and target to be 1D or 2D tensors,"
            f" but received tensors with dimension {preds.shape}"
        )


    # sum_squared_obs = torch.sum(target * target, dim=0)
    residual = target - preds
    rss = torch.sum(residual * residual, dim=0)
    n_obs = target.size(0)
    return rss, n_obs


def _r2_score_compute(
    num_classes: int,
    rss: Tensor,
    n_obs: Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> Tensor:
    """Computes R2 score.

    Args:
        sum_squared_obs: Sum of square of all observations
        rss: Residual sum of squares
        n_obs: Number of predictions or observations
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

    Example:
        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> sum_squared_obs, sum_obs, rss, n_obs = _r2_score_update(preds, target)
        >>> _r2_score_compute(sum_squared_obs, sum_obs, rss, n_obs, multioutput="raw_values")
        tensor([0.8333, 0.96667])
    """
    if n_obs < 2:
        raise ValueError("Needs at least two samples to calculate r2 score.")

    rand_mean_squared_error = _expectation_X_minus_Y_2_(num_classes)
    tss = torch.tensor(n_obs*rand_mean_squared_error)

    device = rss.get_device()

    if device >= 0:
        tss.to(device)

    # Account for near constant targets

    raw_scores = 1 - (rss / tss)

    if multioutput == "raw_values":
        r2 = raw_scores
    elif multioutput == "uniform_average":
        r2 = torch.mean(raw_scores)
    elif multioutput == "variance_weighted":
        tss_sum = torch.sum(tss)
        r2 = torch.sum(tss / tss_sum * raw_scores)
    else:
        raise ValueError(
            "Argument `multioutput` must be either `raw_values`,"
            f" `uniform_average` or `variance_weighted`. Received {multioutput}."
        )

    if adjusted < 0 or not isinstance(adjusted, int):
        raise ValueError("`adjusted` parameter should be an integer larger or" " equal to 0.")

    if adjusted != 0:
        if adjusted > n_obs - 1:
            rank_zero_warn(
                "More independent regressions than data points in"
                " adjusted r2 score. Falls back to standard r2 score.",
                UserWarning,
            )
        elif adjusted == n_obs - 1:
            rank_zero_warn("Division by zero in adjusted r2 score. Falls back to" " standard r2 score.", UserWarning)
        else:
            r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - adjusted - 1)
    return r2


def r2_score_random(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> Tensor:
    r"""Computes r2 random score :

    .. math:: R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\sum_i (y_i_rand-f(x_i)_rand)^2` is total sum of squares. 
    Where y_i_rand is a random class and f(x_i)_rand is also a random class. 
    Therefore y_i_rand and f(x_i)_rand are random discrete variables with a uniform discrete distribution. 
    Calculating SS_tot can be expensive due to number of random number that need to be generated. 
    Thus instead it is calculated based on the expectation value of (y_rand-f(x)_rand).

    Futhermore, it is expected that all class inside the ground truth go from 0 to N-1
    where N is the number of classes.  Additionally all predictions should also be bound and go from 
    0 to N-1. 


    Can also calculate adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the ``adjusted`` argument.

    Args:
        preds: estimated labels
        target: ground truth labels
        num_classses: number of classes that the ground truth can be. 
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Raises:
        ValueError:
            If both ``preds`` and ``targets`` are not ``1D`` or ``2D`` tensors.
        ValueError:
            If ``len(preds)`` is less than ``2`` since at least ``2`` sampels are needed to calculate r2 score.
        ValueError:
            If ``multioutput`` is not one of ``raw_values``, ``uniform_average`` or ``variance_weighted``.
        ValueError:
            If ``adjusted`` is not an ``integer`` greater than ``0``.

    """
    rss, n_obs = _r2_score_update(preds, target)
    return _r2_score_compute(num_classes, rss, n_obs, adjusted, multioutput)


def ratio_rmse(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    mean_squared_error = MeanSquaredError()
    rmse = torch.sqrt(mean_squared_error(preds, target))

    rmse_rand = math.sqrt(_expectation_X_minus_Y_2_(num_classes))
    output = (1 - rmse/rmse_rand)

    return output


def ratio_rmse_random(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    mean_squared_error = MeanSquaredError()
    rmse = torch.sqrt(mean_squared_error(preds, target))
    pred_rand = torch.randint(low=0, high=num_classes, size=preds.shape).float()
    rmse_rand = torch.sqrt(mean_squared_error(pred_rand, target))

    output = (1 - rmse/rmse_rand)

    return output


def absolute_mean_difference(num_classes: int) -> float:

    md = 0
    for x in range(0, num_classes):
        for y in range(0, num_classes):
            md += 1/(num_classes)**2*abs(x-y)

    return md


def _expectation_X_minus_Y_2_(num_classes: int) -> float:

    e_x = (num_classes-1)/2
    e_x2 = (num_classes-1)*(2*num_classes-1)/6
    e_x_minus_y_2 = 2*e_x2 - 2*e_x**2

    return e_x_minus_y_2


def _test_(target, pred, pred_random):
    from torchmetrics.functional import r2_score

    r2 = r2_score(preds=pred.flatten(), target=target.flatten())
    print(f'R2 Score:{r2} ')

    r2 = r2_score(preds=pred_random.flatten(), target=target.flatten())
    print(f'R2 Score with Random prediction:{r2} ')

    r2 = r2_score_random(preds=pred.flatten(), target=target.flatten(), num_classes=11)
    print(f'R2 Score with fixed:{r2} ')

    r2 = r2_score_random(preds=pred_random.flatten(), target=target.flatten(), num_classes=11)
    print(f'R2 Score with fixed & random prediction:{r2} ')

    r2 = ratio_rmse(preds=pred.flatten(), target=target.flatten(), num_classes=11)
    print(f'RMSE ration:{r2} ')

    r2 = ratio_rmse(preds=pred_random.flatten(), target=target.flatten(), num_classes=11)
    print(f'RMSE ration with random prediction:{r2} ')


if __name__ == '__main__':

    # Check the performace of R^2, R^2_fixed, and rmse_ration

    # --------- Test #1 --------- #
    # The first test scene where the ground truth in a random class. 
    # The model predicts correct for 80% of the pixels and 20% it predicts a random class

    print('Test #1 --------------')

    n_classes = 11

    target = torch.randint(0, n_classes, (100, 100)).float()

    pred = target.clone().detach().float()
    pred[0:20, :] = torch.randint(0, n_classes, (20, 100)).float()

    pred_random = torch.randint(0, n_classes, (100, 100)).float()

    _test_(target, pred, pred_random)

    # --------- Test #2 --------- #
    # The scene 80% of  water (class 0), the rest is 10% ice (class 1).  
    # The model predicts correctly precits all the pixels in water but mistakes 10% concertration for 90% concentration. 

    print('Test #2 --------------')

    target = torch.zeros((100, 100))
    target[0:20, :] = 1.0

    pred = torch.zeros((100, 100))
    pred[0:20, :] = 9.0

    pred_random = torch.randint(0, n_classes, (100, 100)).float()

    _test_(target, pred, pred_random)

    # --------- Test #2 --------- #
    # The scene 20% of scene equals 50%,  20% of scene equals 70% and 60% equals 100% ice.  
    # The prediction is wrong about 30% of the scene. Here it predicts water when there is 100% ice.  . 

    print('Test #3 --------------')

    target = torch.zeros((100, 100))
    target[0:20, :] = 5.0
    target[20:40, :] = 7.0
    target[40:, :] = 10.0

    pred = target.clone().detach()
    pred[70:, :] = 0.0

    pred_random = torch.randint(0, n_classes, (100, 100)).float()

    _test_(target, pred, pred_random)

    # Testing the perfomace of function _expectation_X_minus_Y_2_
    print('Testing the function _expectation_X_minus_Y_2_')
    print('------------------------------------------------')
    n = 11
    a = torch.randint(0, n, (10000, 100000)).float()
    b = torch.randint(0, n, (10000, 100000)).float()

    c = (a-b) ** 2

    print(f'Number of classes:{n}')
    print(f'Experimental expectation of (x-y)^2:{c.mean()}')

    e_x = (n-1)/2
    e_x2 = (n-1)*(2*n-1)/6
    e_x_min_y_2 = _expectation_X_minus_Y_2_(n)

    print(f'Theoritical expectation of (x-y)^2:{e_x_min_y_2}')

    print('------------------------------------------------')
    n = 12
    a = torch.randint(0, n, (10000, 100000)).float()
    b = torch.randint(0, n, (10000, 100000)).float()

    c = (a-b) ** 2

    print(f'Number of classes:{n}')
    print(f'Experimental expectation of (x-y)^2:{c.mean()}')

    e_x = (n-1)/2
    e_x2 = (n-1)*(2*n-1)/6
    e_x_min_y_2 = _expectation_X_minus_Y_2_(n)

    print(f'Theoritical expectation of (x-y)^2:{e_x_min_y_2}')

    # --------------------------------------------------------
    # Testing R^2 random
    print('Testing R^2 random function')

    target = torch.tensor([[0, 1], [3, 1], [7, 5]])
    preds = torch.tensor([[0, 2], [2, 2], [4, 5]])
    
    print(r2_score_random(preds, target, num_classes=11, multioutput="raw_values"))


    target = torch.tensor([[0, 1], [3, 1], [7, 5]])
    preds = torch.tensor([[0, 1], [3, 1], [7, 5]])
    
    print(r2_score_random(preds, target, num_classes=11, multioutput="raw_values"))


