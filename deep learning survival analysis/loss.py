from typing import Sequence, Optional
import tensorflow as tf
import numpy as np

def safe_normalize(x: tf.Tensor) -> tf.Tensor:
    """
    Normalize risk scores to avoid exp underflowing.
    Note that only risk scores relative to each other matter.
    If minimum risk score is negative, we shift scores so minimum
    is at zero.
    """
    x_min = tf.reduce_min(x, axis=0) # keep the minimum values along the 0 axis (rows)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm

def logsumexp_masked(risk_scores: tf.Tensor,
                     mask: tf.Tensor,
                     axis: int = 0,
                     keepdims: Optional[bool] = None) -> tf.Tensor:
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)
        # for numerical stability, substract the maximum value
        # before taking the exponential
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax

        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output

def _make_riskset(time: np.ndarray) -> np.ndarray:
    """Compute mask that represents each sample's risk set.
    Parameters
    ----------
    time : np.ndarray, shape=(n_samples,)
        Observed event time sorted in descending order.
    Returns
    -------
    risk_set : np.ndarray, shape=(n_samples, n_samples)
        Boolean matrix where the `i`-th row denotes the
        risk set of the `i`-th instance, i.e. the indices `j`
        for which the observer time `y_j >= y_i`.
    """
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    riskset = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        riskset[i_sort, o[:k]] = True

    if riskset.ndim != 2:
        raise ValueError("Rank mismatch: Rank of riskset should be 2")
    return riskset


class CoxPHLoss(tf.keras.losses.Loss):
    """ Negative partial log-likelihood of Cox's proportional hazards model. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)            

    def call(self, y_true: Sequence[tf.Tensor], predictions: tf.Tensor) -> tf.Tensor:
        """
        Compute loss function .
        Parameters
        ----------
        event:
        time:
        predictions:
        y_true : list|tuple of tf.Tensor
            The first element holds a binary vector where 1 indicates an event and 0 censoring.
            The second element holds the riskset, a boolean matrix where the `i`-th row denotes the
            risk set of the `i`-th instance, i.e. the indices `j` for which the observer time `y_j >= y_i`.
            Both must be rank 2 tensors.
        y_pred : tf.Tensor
            The predicted outputs. Must be a rank 2 tensor.
        Returns
        -------
        loss : tf.Tensor
            Loss for each instance in the batch.
        """
        # print('event', y_true[0])
        # print('time', y_true[1])
        event, time = y_true   
        event = tf.reshape(event, predictions.shape)
        riskset = _make_riskset(time.numpy())
        if predictions.shape.ndims != 2:
            raise ValueError("Rank mismatch: Rank of predictions should be 2")

        if predictions.shape[1] is None:
            raise ValueError("Last dimension of predictions must be known")

        if predictions.shape[1] != 1:
            raise ValueError("Dimension mismatch: Last dimension of predictions must be 1")

        if event.shape.ndims != predictions.shape.ndims:
            raise ValueError("Rank mismatch: Rank of predictions & rank of event should be equal")

        event = tf.cast(event, predictions.dtype)
        predictions = safe_normalize(predictions)

        with tf.name_scope("assertions"):
            assertions = (
                tf.debugging.assert_less_equal(event, 1.),
                tf.debugging.assert_greater_equal(event, 0.),
                tf.debugging.assert_type(riskset, tf.bool)
            )

        # move batch dimension to the end so predictions get broadcast
        # row-wise when multiplying by riskset
        predictions_t = tf.transpose(predictions)
        # compute log of sum over risk set for each row
        rr = logsumexp_masked(predictions_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()

        losses = tf.math.multiply(event, rr - predictions)

        return losses