import argparse
import pytorch_lightning
import os
import json
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import easyopt

from src.data.dataloaders import train_dataloader, val_dataloader, test_dataloader
from src.data.dataset import WaterDataset

from src.models.autoencoder import WaterAutoEncoder
from src.models.regression import WaterRegressionModel
from test import test, make_plots

def train(args):
    loggers = list()
    callbacks = list()
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(entity=args.wandb_entity, project=args.wandb_project)
        loggers.append(wandb_logger)

    early_stop_callback = EarlyStopping(
        monitor="validation/loss",
        min_delta=0.00,
        patience=args.patience,
        mode="min"
    )
    callbacks.append(early_stop_callback)

    ds = WaterDataset(os.path.join(args.dataset, "train.pkl"), normalize=args.normalize)

    vars(args)["input_channels"] = ds.input_channels
    vars(args)["output_channels"] = ds.output_channels
    vars(args)["y_mean"] = ds.y_mean
    vars(args)["y_std"] = ds.y_std

    model = globals()[args.model](args)

    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        callbacks=callbacks,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        fast_dev_run=args.dry_run,
    )
    
    trainer.fit(model, train_dataloader(args, normalize=args.normalize), val_dataloader(args, normalize=args.normalize))
    
    train_results = test(model, train_dataloader(args, normalize=False))
    validation_results = test(model, val_dataloader(args, normalize=False))
    test_results = test(model, test_dataloader(args, normalize=False))

    results = dict(
        train=train_results,
        validation=validation_results,
        test=test_results
    )

    print(json.dumps(results, indent=4))

    easyopt.objective(results["validation"]["overall_mean_absolute_error"])

    if args.wandb:
        import wandb
        wandb.log(dict(results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="./dataset/dataset")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--model", type=str, default="WaterRegressionModel", choices=["WaterRegressionModel", "WaterAutoEncoder"])
    parser.add_argument("--do-not-normalize", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--loss", type=str, default="MSELoss", choices=["MSELoss", "L1Loss"])
    parser.add_argument("--features-size", type=int, default=16)
    parser.add_argument("--encoder-l1-channels", type=int, default=5)
    parser.add_argument("--encoder-l2-channels", type=int, default=10)
    parser.add_argument("--decoder-l1-channels", type=int, default=5)
    parser.add_argument("--decoder-l2-channels", type=int, default=10)
    
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=300)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="water-manel")
    parser.add_argument("--wandb-tag", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="mlpi")
    parser.add_argument("--log-images", action="store_true")

    parser.add_argument("--study-name", type=str, default="")
    
    args = parser.parse_args()

    vars(args)["normalize"] = not args.do_not_normalize
    
    if args.seed < 0:
        random_data = os.urandom(4)
        args.seed = int.from_bytes(random_data, byteorder="big")

    seed_everything(args.seed)
    train(args)
