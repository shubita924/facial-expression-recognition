import argparse, random
import numpy as np
import torch
import torch.nn as nn
import wandb

from dataset import FER2013Dataset
from models import BaselineMLP
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--data", default="data/fer2013.csv")
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(FER2013Dataset(args.data, "train"),
                              batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(FER2013Dataset(args.data, "val"),
                            batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BaselineMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    name = f"baseline_mlp_lr{args.lr}_bs{args.batch_size}"
    run = wandb.init(project="fer2013", name=name, config=vars(args))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        run.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                 "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{epoch:2d}/{args.epochs}] train_acc={tr_acc:.3f} val_acc={val_acc:.3f} "
              f"| train_loss={tr_loss:.3f} val_loss={val_loss:.3f}")

    run.finish()

if __name__ == "__main__":
    main()
