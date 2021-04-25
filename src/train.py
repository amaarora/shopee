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
from accelerate import Accelerator
logging.getLogger().setLevel(logging.INFO)


def train_fn(model, data_loader, optimizer, scheduler, i, accelerator):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Epoch" + " [TRAIN] " + str(i + 1))

    for t, data in enumerate(tk):
        optimizer.zero_grad()
        out = model(**data)
        loss = nn.CrossEntropyLoss()(out, data["label"])
        accelerator.backward(loss)
        optimizer.step()
        fin_loss += loss.item()

        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )

    scheduler.step()

    return fin_loss / len(data_loader), optimizer.param_groups[0]["lr"]


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


def train():
    # initialize accelerator
    accelerator = Accelerator()

    # wandb init
    wandb.init(project="shopee", config=Config)

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

    # optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config["SCHEDULER_PARAMS"]["lr_start"]
    )
    scheduler = ShopeeScheduler(optimizer, **Config["SCHEDULER_PARAMS"])

    # prepare for DDP
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    for epoch in range(Config["EPOCHS"]):
        avg_loss_train, lr = train_fn(
            model, train_dataloader, optimizer, scheduler, epoch, accelerator
        )
        wandb.log({'train_loss': avg_loss_train, 'epoch': epoch, 'lr': lr})


if __name__ == "__main__":
    train()