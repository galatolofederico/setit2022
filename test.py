import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import json
from tqdm import tqdm

from src.data.dataloaders import train_dataloader, val_dataloader, test_dataloader
from src.data.dataset import WaterDataset

from src.models.autoencoder import WaterAutoEncoder
from src.models.regression import WaterRegressionModel

def make_plots(model, data_loader, output_folder=None, wandb=False):
    model.eval()
    for batch_nb, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_X, batch_y, batch_mask = batch
        batch_y_hat = model(batch_X, batch_mask)

        batch_X = batch_X.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        batch_y_hat = batch_y_hat.detach().cpu().numpy()
        for elem_number, (X, y, y_hat) in tqdm(enumerate(zip(batch_X, batch_y, batch_y_hat)), total=batch_X.shape[0]):
            fig, axes = plt.subplots(4, max(X.shape[0], y.shape[0]), figsize=(30, 30))

            absolute_error = np.abs(y_hat - y)
            for i, (ax, data) in enumerate(zip(axes[0, :], X)):
                ax.set_title(f"input.{i}")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                img = ax.matshow(data)
                fig.colorbar(img, cax=cax, orientation='vertical')

            for i, (ax, data) in enumerate(zip(axes[1, :], y_hat)):
                ax.set_title(f"output.{i}")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                img = ax.matshow(data)
                fig.colorbar(img, cax=cax, orientation='vertical')

            for i, (ax, data) in enumerate(zip(axes[2, :], y)):
                ax.set_title(f"target.{i}")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                img = ax.matshow(data)
                fig.colorbar(img, cax=cax, orientation='vertical')

            for i, (ax, data) in enumerate(zip(axes[3, :], absolute_error)):
                ax.set_title(f"absolute_error.{i}")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                img = ax.matshow(data)
                fig.colorbar(img, cax=cax, orientation='vertical')

            if output_folder is not None:
                fig.savefig(os.path.join(output_folder, f"input_output_{batch_nb}_{elem_number}.png"))



def test(model, data_loader):
    model.eval()
    stats = dict()
    for batch in data_loader:
        batch_X, batch_y, batch_mask = batch
        batch_y_hat = model(batch_X, batch_mask)

        batch_X = batch_X.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        batch_y_hat = batch_y_hat.detach().cpu().numpy()
        
        for X, y, y_hat in zip(batch_X, batch_y, batch_y_hat):
            absolute_error = np.abs(y_hat - y)

            for channel, aa in enumerate(absolute_error):
                if f"output_{channel}_mean_absolute_error" not in stats:
                    stats[f"output_{channel}_mean_absolute_error"] = list()       
                stats[f"output_{channel}_mean_absolute_error"].append(aa.mean())
        
            if f"overall_mean_absolute_error" not in stats:
                stats[f"overall_mean_absolute_error"] = list()

            stats[f"overall_mean_absolute_error"].append(absolute_error.mean())
        
        mean_stats = dict()
        for k, v in stats.items():
            mean_stats[k] = sum(v)/len(v)
        
        return mean_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=["WaterRegressionModel", "WaterAutoEncoder"], default="WaterRegressionModel")
    parser.add_argument("--dataset", type=str, default="test", choices=["train", "test", "validation"], dest="dataset_type")
    parser.add_argument("--dataset-folder", type=str, default="./dataset/dataset", dest="dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-folder", type=str, default="plots")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--make-results", action="store_true")
    

    args = parser.parse_args()

    model_class = globals()[args.model_type]
    model = model_class.load_from_checkpoint(args.model)

    data_loader = None

    if args.dataset_type == "train":
        data_loader = train_dataloader(args, normalize=False)
    elif args.dataset_type == "test":
        data_loader = test_dataloader(args, normalize=False)
    elif args.dataset_type == "validation":
        data_loader = val_dataloader(args, normalize=False)
    else:
        raise Exception(f"Unknown dataset {args.dataset_type}")
    
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    output_folder = os.path.join(args.output_folder, args.dataset_type)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if args.make_plots:
        make_plots(model, data_loader, output_folder)
    
    if args.make_results:
        results = test(model, data_loader)
        print(json.dumps(results, indent=4))
        json.dump(results, open(os.path.join(output_folder, "results.json"), "w"), indent=4)