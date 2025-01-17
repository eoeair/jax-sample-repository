import jax
import jax.numpy as jnp
from utils import metric


class ConfusionMatrix(metric.Metric):
    """
    Constructs a confusion matrix for multi-class classification problems.
    Does not support multi-label.
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()
        self.conf = jax.device_put(
            jnp.zeros((num_classes, num_classes), dtype=jnp.int64),
            device=jax.devices("cpu")[0],
        )
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf = jax.device_put(
            jnp.zeros((self.num_classes, self.num_classes), dtype=jnp.int64),
            device=jax.devices("cpu")[0],
        )

    def add(self, predicted, target):
        # 如果是 jax 数组，则转为 np.array
        if isinstance(predicted, jnp.ndarray):
            predicted = jnp.array(predicted)
        if isinstance(target, jnp.ndarray):
            target = jnp.array(target)

        assert (
            predicted.shape[0] == target.shape[0]
        ), "number of targets and predicted outputs do not match"

        # 如果 predicted 不是 1D，则取 argmax
        if predicted.ndim != 1:
            assert (
                predicted.shape[1] == self.num_classes
            ), "number of predictions does not match size of confusion matrix"
            predicted = jnp.argmax(predicted, axis=1)
        else:
            assert (
                0 <= predicted.min() and predicted.max() < self.num_classes
            ), "predicted values are not between 0 and k-1"

        # 如果 target 不是 1D，则取 argmax
        if target.ndim != 1:
            assert (
                target.shape[1] == self.num_classes
            ), "Onehot target does not match size of confusion matrix"
            assert (target >= 0).all() and (
                target <= 1
            ).all(), "in one-hot encoding, target values should be 0 or 1"
            assert (
                target.sum(axis=1) == 1
            ).all(), "multi-label setting is not supported"
            target = jnp.argmax(target, axis=1)
        else:
            assert (
                0 <= target.min() and target.max() < self.num_classes
            ), "target values are not between 0 and k-1"

        # 构建混淆矩阵
        x = predicted + self.num_classes * target
        bincount_2d = jnp.bincount(x.astype(jnp.int64), minlength=self.num_classes**2)
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        self.conf += conf

    def value(self):
        if self.normalized:
            conf = self.conf.astype(jnp.float32)
            return conf / jnp.clip(conf.sum(axis=1), a_min=1e-12)[:, None]
        else:
            return self.conf
