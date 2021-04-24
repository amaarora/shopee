from config import Config
from dataset import Shopee_Dataset
from model import Shopee_Model
from scheduler import ShopeeScheduler
from sklearn.preprocessing import LabelEncoder
import cuml
import torch
import wandb
import torch.nn as nn
import os
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import pandas as pd
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def train_fn(model, data_loader, optimizer, scheduler, i, device_ids):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Epoch" + " [TRAIN] " + str(i + 1))

    for t, data in enumerate(tk):
        for k, v in data.items():
            data[k] = v.to(device_ids[0])
        optimizer.zero_grad()
        out = model(**data)
        loss = nn.CrossEntropyLoss()(out, data["label"])
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )

    scheduler.step()

    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, i):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Epoch" + " [VALID] " + str(i + 1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            for k, v in data.items():
                data[k] = v.to(Config["DEVICE"])
            out = model(**data)
            loss = nn.CrossEntropyLoss()(out, data["label"])
            fin_loss += loss.item()

            tk.set_postfix({"loss": "%.6f" % float(fin_loss / (t + 1))})
        return fin_loss / len(data_loader)


def train(local_world_size, local_rank):
    # setup devices for this process. For local_world_size = 2, num_gpus = 8,
    # rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )
    # train data
    labelencoder = LabelEncoder()
    train_df = pd.read_csv(Config["TRAIN_CSV_PATH"])
    train_df["label_group"] = labelencoder.fit_transform(train_df["label_group"])
    train_df["img_path"] = Config["TRAIN_DATA_DIR"] + "/" + train_df.image

    NUM_CLASS = len(set(train_df.label_group))
    TRAIN_IMG_PATHS = train_df.img_path.values
    TRAIN_LABELS = train_df.label_group.values

    # train dataset & dataloader
    train_dataset = Shopee_Dataset(
        TRAIN_IMG_PATHS, TRAIN_LABELS, augmentations=Config["TRAIN_AUG"]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config["BS"]
    )

    # model
    model = Shopee_Model(Config["MODEL"], num_class=NUM_CLASS, pretrained=True)
    model.backbone.reset_classifier(0)
    model = model.cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)

    # optimizer & scheduler
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=Config["SCHEDULER_PARAMS"]["lr_start"]
    )
    scheduler = ShopeeScheduler(optimizer, **Config["SCHEDULER_PARAMS"])

    for epoch in range(Config["EPOCHS"]):
        avg_loss_train = train_fn(
            ddp_model, train_dataloader, optimizer, scheduler, epoch, device_ids
        )


def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    train(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    spmd_main(args.local_world_size, args.local_rank)
