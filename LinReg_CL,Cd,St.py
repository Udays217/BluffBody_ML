import argparse
import csv
from pathlib import Path

import torch
from torch import nn


INPUT_COLUMN = "Reynolds Number"
OUTPUT_COLUMNS = ["C_L Amplitude", "Mean C_D", "Strouhal Number"]
DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "Results.csv"
MODEL_PATH = Path(__file__).resolve().parent / "linear_regression_model.pt"


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 3)

    def forward(self, x):
        return self.linear(x)


def load_dataset(csv_path):
    rows = []

    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(row for row in file if row.strip().strip(","))
        for row in reader:
            try:
                x_value = float(row[INPUT_COLUMN])
                y_values = [float(row[column]) for column in OUTPUT_COLUMNS]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((x_value, y_values))

    if not rows:
        raise ValueError(f"No usable training rows found in {csv_path}")

    x = torch.tensor([[row[0]] for row in rows], dtype=torch.float32)
    y = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    return x, y


def normalize(value, mean, std):
    return (value - mean) / std


def denormalize(value, mean, std):
    return value * std + mean


def train_model(x, y, epochs=5000, learning_rate=0.05):
    torch.manual_seed(42)

    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0)

    x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
    y_std = torch.where(y_std == 0, torch.ones_like(y_std), y_std)

    x_train = normalize(x, x_mean, x_std)
    y_train = normalize(y, y_mean, y_std)

    model = LinearRegressionModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        predictions = model(x_train)
        loss = loss_function(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1:5d} | Loss: {loss.item():.8f}")

    metadata = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "output_columns": OUTPUT_COLUMNS,
    }
    return model, metadata


def predict(model, metadata, reynolds_number):
    model.eval()
    x_value = torch.tensor([[reynolds_number]], dtype=torch.float32)
    x_value = normalize(x_value, metadata["x_mean"], metadata["x_std"])

    with torch.no_grad():
        y_value = model(x_value)

    y_value = denormalize(y_value, metadata["y_mean"], metadata["y_std"])
    return y_value.squeeze(0)


def save_model(model, metadata):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        MODEL_PATH,
    )


def load_saved_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = LinearRegressionModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["metadata"]


def print_prediction(values):
    for column_name, value in zip(OUTPUT_COLUMNS, values):
        print(f"{column_name}: {value.item():.6f}")


def print_equations(model, metadata):
    weights = model.linear.weight.detach()
    bias = model.linear.bias.detach()

    x_mean = metadata["x_mean"].item()
    x_std = metadata["x_std"].item()
    y_mean = metadata["y_mean"]
    y_std = metadata["y_std"]

    print("\nFitted linear equations:")
    for index, column_name in enumerate(OUTPUT_COLUMNS):
        slope = (weights[index, 0] * y_std[index] / x_std).item()
        intercept = (y_mean[index] + y_std[index] * bias[index] - slope * x_mean).item()
        print(f"{column_name} = {slope:.8f} * Reynolds Number + {intercept:.8f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a PyTorch linear regression model for Reynolds-number outputs."
    )
    parser.add_argument(
        "--reynolds",
        "-r",
        type=float,
        help="Reynolds number to predict after training/loading the model.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the training CSV. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the saved model instead of training again.",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.load:
        model, metadata = load_saved_model()
    else:
        x, y = load_dataset(args.csv)
        model, metadata = train_model(x, y, args.epochs, args.learning_rate)
        save_model(model, metadata)
        print(f"\nSaved trained model to: {MODEL_PATH}")
        print_equations(model, metadata)

    if args.reynolds is not None:
        print(f"\nPrediction for Reynolds Number = {args.reynolds:g}")
        print_prediction(predict(model, metadata, args.reynolds))


if __name__ == "__main__":
    main()
