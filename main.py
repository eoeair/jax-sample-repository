import os

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
# )

import yaml

import numpy as np

# torch, jax, flax, optax, orbax
import torch  # just for data loader
import jax
import jax.numpy as jnp
import flax.serialization
import flax.nnx as nn
import optax
import orbax.checkpoint as orbax

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
    device = jax.devices(arg.device)[0]

    # set up tensorboard
    writer = SummaryWriter(arg.work_dir)

    # set work folder
    if not os.path.exists(arg.work_dir):
        os.makedirs(arg.work_dir)

    # set up data loader
    Feeder = import_class(arg.feeder)
    if arg.phase == "train":
        trainloader = torch.utils.data.DataLoader(
            dataset=Feeder(**arg.train_feeder_args),
            batch_size=arg.batch_size,
            shuffle=True,
            num_workers=arg.num_worker,
            pin_memory=True,
        )
    testloader = torch.utils.data.DataLoader(
        dataset=Feeder(**arg.test_feeder_args),
        batch_size=arg.test_batch_size,
        shuffle=False,
        num_workers=arg.num_worker,
        pin_memory=True,
    )

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
    path = os.path.join(arg.work_dir, "ckpt")
    path = os.path.abspath(path)  # 获取绝对路径
    checkpointer = orbax.PyTreeCheckpointer()
    if os.path.exists(path):
        ckpt = checkpointer.restore(f"{path}")
        state = ckpt["state"]
        arg.start_epoch = ckpt["start_epoch"]

    nn.update((model, optimizer), state)

    # one more metric
    miou = IoU(arg.model_args["num_class"])

    @nn.jit
    def train_step(model, optimizer: nn.Optimizer, data, label):

        def loss_fn(model):
            logits = model(data)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=label
            ).mean()
            return loss

        loss, grads = nn.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

    @nn.jit
    def test_step(model, data, label):
        logits = model(data)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=label
        ).mean()
        return loss, logits

    # Executes a training loop.
    for epoch in range(arg.start_epoch, arg.num_epoch):
        print("Epoch:[{}/{}]".format(epoch + 1, arg.num_epoch))
        best_test_miou = 0

        for batch_idx, (data, label) in enumerate(tqdm(trainloader)):
            data = jnp.array(data, dtype=jnp.bfloat16, device=device)
            label = jnp.array(label, dtype=jnp.int32, device=device)
            loss = train_step(model, optimizer, data, label)
            writer.add_scalar("loss", loss, epoch * iter_per_epoch_train + batch_idx)
            if epoch == arg.num_epoch - 1 and batch_idx == len(trainloader) - 1:
                state = nn.state((model, optimizer))
                checkpointer = orbax.PyTreeCheckpointer()
                ckpt = {"state": state, "start_epoch": epoch + 1}
                checkpointer.save(f"{path}", ckpt)

        print("loss: {}".format(loss))

        if epoch % 10 == 0:
            for batch_idx, (data, label) in enumerate(tqdm(testloader)):
                data = jnp.array(data, dtype=jnp.bfloat16, device=device)
                label_jax = jnp.array(label, dtype=jnp.int32, device=device)
                miou.reset()
                loss, logits = test_step(model, data, label_jax)
                # compute iou and miou
                logits = jnp.array(logits, device=jax.devices("cpu")[0])
                miou.add(logits, label)
                iou, m_iou = miou.value()
                print("Test iou: {} Test miou: {} loss: {}".format(iou, miou, loss))
                if m_iou > best_test_miou:
                    best_test_miou = m_iou

    print("Best Test miou: {}".format(best_test_miou))
