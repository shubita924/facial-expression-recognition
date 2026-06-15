import argparse, os, sys, random, inspect
import numpy as np
import torch
import torch.nn as nn
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FER2013Dataset
from models import get_model, REGISTRY
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(name, dropout, batchnorm):
    """Pass only the args each model's constructor actually accepts."""
    cls = REGISTRY[name]
    params = inspect.signature(cls.__init__).parameters
    kwargs = {}
    if "dropout" in params:   kwargs["dropout"] = dropout
    if "batchnorm" in params: kwargs["batchnorm"] = batchnorm
    return cls(**kwargs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline_mlp")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batchnorm", action="store_true")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0,
                   help="early stop if val_acc doesn't improve for N epochs (0 = off)")
    p.add_argument("--data", default="data/fer2013.csv")
    p.add_argument("--out_dir", default="checkpoints")
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader = DataLoader(FER2013Dataset(args.data, "train"),
                              batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(FER2013Dataset(args.data, "val"),
                            batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(FER2013Dataset(args.data, "test"),
                             batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model, args.dropout, args.batchnorm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    bn = "_bn" if args.batchnorm else ""
    name = f"{args.model}_lr{args.lr}_bs{args.batch_size}_do{args.dropout}_wd{args.weight_decay}{bn}"
    run = wandb.init(project="fer2013", name=name, config=vars(args))

    ckpt_path = os.path.join(args.out_dir, f"{name}.pt")
    best_val_acc, best_epoch, epochs_no_improve = 0.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        run.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                 "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{epoch:2d}/{args.epochs}] train_acc={tr_acc:.3f} val_acc={val_acc:.3f} "
              f"| train_loss={tr_loss:.3f} val_loss={val_loss:.3f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, epochs_no_improve = val_acc, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stop at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

    model.load_state_dict(torch.load(ckpt_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    run.summary["best_val_acc"] = best_val_acc
    run.summary["best_epoch"] = best_epoch
    run.summary["test_acc"] = test_acc
    print(f"DONE  best_val_acc={best_val_acc:.3f} @epoch {best_epoch}  test_acc={test_acc:.3f}")
    run.finish()

if __name__ == "__main__":
    main()
