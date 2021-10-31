import argparse
import pickle
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--input", default="./dataset/dataset/dataset.pkl")
parser.add_argument("--output-folder", default="./dataset/dataset/")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-perc", type=float, default=0.2)
parser.add_argument("-val-perc", type=float, default=0.25)

args = parser.parse_args()

dataset = pickle.load(open(args.input, "rb"))
X = dataset["X"]
y = dataset["y"]
mask = dataset["mask"]

X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(X, y, mask, test_size=args.test_perc, random_state=args.seed)
X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(X_train, y_train, mask_train, test_size=args.val_perc, random_state=args.seed)

print(f"Train size:\t{len(X_train)}")
print(f"Val size:\t{len(X_val)}")
print(f"Test size:\t{len(X_test)}")

pickle.dump(
    dict(
        X=X_train,
        y=y_train,
        mask=mask_train
    ),
    open(os.path.join(args.output_folder, "train.pkl"), "wb")
)

pickle.dump(
    dict(
        X=X_val,
        y=y_val,
        mask=mask_val
    ),
    open(os.path.join(args.output_folder, "val.pkl"), "wb")
)

pickle.dump(
    dict(
        X=X_test,
        y=y_test,
        mask=mask_test
    ),
    open(os.path.join(args.output_folder, "test.pkl"), "wb")
)