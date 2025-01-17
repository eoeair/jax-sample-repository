import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
)

import yaml

import numpy as np

# jax, flax, optax, tensorflow
import jax
import flax.serialization
import flax.nnx as nn
import optax
import tensorflow as tf

# record
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.parser import get_parser
from utils.iou import IoU
from utils.utils import import_class


def save_arg(arg):
    # save arg
    arg_dict = vars(arg)
    if not os.path.exists(arg.work_dir):
        os.makedirs(arg.work_dir)
    with open("{}/config.yaml".format(arg.work_dir), "w") as f:
        yaml.dump(arg_dict, f)


if __name__ == "__main__":
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    # set device
    # device = jax.devices()[arg.device]

    # set up tensorboard
    writer = SummaryWriter(arg.work_dir)

    # set work folder
    if not os.path.exists(arg.work_dir):
        os.makedirs(arg.work_dir)

    # set up data loader
    Feeder = import_class(arg.feeder)

    if arg.phase == "train":
        # 调用 get_dataset() 获取 tf.data.Dataset
        train_dataset = Feeder(**arg.train_feeder_args).get_dataset()
        train_dataset = (
            train_dataset.shuffle(buffer_size=arg.buffer_size)
            .batch(arg.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        trainloader = train_dataset

    test_dataset = Feeder(**arg.test_feeder_args).get_dataset()
    test_dataset = test_dataset.batch(arg.test_batch_size).prefetch(tf.data.AUTOTUNE)
    testloader = test_dataset

    # set up model
    Model = import_class(arg.model)
    model = Model(**arg.model_args, train=True, seed=arg.seed)

    # set up lr_schedule and optimizer
    # TODO: 数字获取方式
    iter_per_epoch_train = 3630 // arg.batch_size
    lr_schedule = optax.linear_onecycle_schedule(
        arg.num_epoch * iter_per_epoch_train, arg.base_lr
    )
    if arg.optimizer == "SGD":
        optimizer = optax.sgd(lr_schedule, momentum=0.9, nesterov=arg.nesterov)
    elif arg.optimizer == "Adam":
        optimizer = optax.adam(lr_schedule, nesterov=arg.nesterov)
    else:
        raise ValueError("Unknown optimizer: {}".format(arg.optimizer))

    # Defines parameter update function.
    optimizer = nn.Optimizer(model, optimizer)

    state = nn.state((model, optimizer))

    # automatically resume from checkpoint if it exists
    path = os.path.join(arg.work_dir, "checkpoint.jax")
    if os.path.exists(path):
        with open(path, "rb") as f:
            byte_data = f.read()
        flax.serialization.from_bytes(
            {"state": state, "start_epoch": arg.start_epoch},
            byte_data,
        )
    else:
        arg.start_epoch = 0

    nn.update((model, optimizer), state)

    # one more metric
    miou = IoU(arg.model_args["num_class"])

    @nn.jit
    def train_step(model, optimizer: nn.Optimizer, batch):
        data, label = batch["data"], batch["label"]

        def loss_fn(model):
            logits = model(data)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=label
            ).mean()
            return loss

        loss, grads = nn.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        print(type(loss))
        print("Train loss: {}".format(loss))

        return loss

    @nn.jit
    def test_step(model, batch):
        data, label = batch["data"], batch["label"]
        logits = model(data)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=label
        ).mean()
        return loss, logits

    # Executes a training loop.
    for epoch in range(arg.start_epoch, arg.num_epoch):
        best_test_miou = 0

        for batch_idx, train_batch in enumerate(
            tqdm(
                trainloader.as_numpy_iterator(), desc=f"Epoch {epoch+1}/{arg.num_epoch}"
            )
        ):
            loss = train_step(model, optimizer, train_batch)
            writer.add_scalar("loss", loss, epoch * iter_per_epoch_train + batch_idx)
            print("loss: {}".format(loss))
            state = nn.state((model, optimizer))
            ckpt_data = {"state": state, "start_epoch": epoch + 1}
            with open(path, "wb") as f:
                f.write(flax.serialization.to_bytes(ckpt_data))

        if epoch % 10 == 0:
            for batch_idx, test_batch in enumerate(
                tqdm(testloader.as_numpy_iterator(), desc="Testing")
            ):
                miou.reset()
                loss, logits = test_step(model, test_batch)
                # compute iou and miou
                label = test_batch["label"]
                miou.add(logits, label)
                iou, m_iou = miou.value()
                print("Test iou: {} Test miou: {} loss: {}".format(iou, miou, loss))
                if m_iou > best_test_miou:
                    best_test_miou = m_iou

    print("Best Test miou: {}".format(best_test_miou))
