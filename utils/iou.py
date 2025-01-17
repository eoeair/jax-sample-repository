import jax
import jax.numpy as jnp
from utils import metric
from utils.confusionmatrix import ConfusionMatrix


class IoU(metric.Metric):
    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        assert (
            predicted.shape[0] == target.shape[0]
        ), "number of targets and predicted outputs do not match"
        assert predicted.ndim == 4, "predictions must be of shape (N, H, W, K)"
        assert (
            target.ndim == 3 or target.ndim == 4
        ), "targets must be of shape (N, H, W) or (N, K, H, W)"

        # 如果是 (N, H, W, K)，转换到 (N, H, W)
        if predicted.ndim == 4:
            predicted = predicted.argmax(axis=-1)
        if target.ndim == 4:
            target = target.argmax(axis=1)

        self.conf_metric.add(predicted.reshape(-1), target.reshape(-1))

    def value(self):
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix = conf_matrix.at[:, self.ignore_index].set(0)
            conf_matrix = conf_matrix.at[self.ignore_index, :].set(0)
        true_positive = jnp.diag(conf_matrix)
        false_positive = jnp.sum(conf_matrix, 0) - true_positive
        false_negative = jnp.sum(conf_matrix, 1) - true_positive

        # 计算 IoU
        with jnp.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (true_positive + false_positive + false_negative)
        return iou, jnp.nanmean(iou)


class IoU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = jax.device_put(
            jnp.zeros((num_classes, num_classes)), device=jax.devices("cpu")[0]
        )

    def reset(self):
        self.hist = jax.device_put(
            jnp.zeros((self.num_classes, self.num_classes)),
            device=jax.devices("cpu")[0],
        )

    def add(self, pred, label):
        if pred.ndim == 4:
            pred = pred.argmax(axis=1)
        if label.ndim == 4:
            label = label.argmax(axis=1)
        if len(label.flatten()) != len(pred.flatten()):
            print(
                f"Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}"
            )
            return

        self.hist += self.fast_hist(label.flatten(), pred.flatten(), self.num_classes)

    def value(self):
        mIoUs = self.per_class_iu(self.hist)
        mPA = self.per_class_PA(self.hist)
        miou = jnp.nanmean(mIoUs)
        return mIoUs, miou

    @staticmethod
    def fast_hist(a, b, n):
        a = jnp.array(a) if isinstance(a, jnp.ndarray) else a
        b = jnp.array(b) if isinstance(b, jnp.ndarray) else b
        k = (a >= 0) & (a < n)
        return jnp.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    @staticmethod
    def per_class_iu(hist):
        return jnp.diag(hist)[1:] / jnp.maximum(
            (hist.sum(1) + hist.sum(0) - jnp.diag(hist))[1:], 1
        )

    @staticmethod
    def per_class_PA(hist):
        return jnp.diag(hist) / jnp.maximum(hist.sum(1), 1)
