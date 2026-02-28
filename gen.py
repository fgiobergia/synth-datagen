import torch
import torch.nn as nn

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

import argparse

def get_model_performance(X, y, model, X_test=None, y_test=None, test_size=0.2, random_state=42):

    if X_test is None or y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        X_train, y_train = X, y

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

class UnderlyingProcess(nn.Module):
    def __init__(self, n_attr, hidden_size=32, n_out=1):
        super(UnderlyingProcess, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_attr, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out)
        )

        # freeze encoder/edecoder
        # NOTE: currently not frozen, but optimizer has lr = 0
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.decoder.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_best_linear_model(X, y):
    """ provide closed-form solution for best linear model (OLS) """
    return (X.T @ X).inverse() @ (X.T @ y)

def get_best_linear_model_predictions(X, y):
    """ get predictions of best linear model (OLS) """
    
    # add bias term to dataset
    X_bias = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    weights = get_best_linear_model(X_bias, y)
    return X_bias @ weights


def loss_func(y_pred_model, y_pred_lin, y_true, lmbda):
    mse_loss = nn.MSELoss()
    model_loss = mse_loss(y_pred_model, y_true)
    lin_loss = -mse_loss(y_pred_lin, y_true)
    total_loss = model_loss + lmbda * lin_loss
    return total_loss

def gen_Xy(n_pts, n_attr, y_distrib, device, n_outs=1, random_state=42):

    rng = np.random.default_rng(random_state)
    X = torch.tensor(rng.random(size=(n_pts, n_attr)), dtype=torch.float32, requires_grad=True, device=device)

    tot_n_pts = n_pts * n_outs

    if y_distrib == "bimodal":
        # bimodal distribution
        # pick loc and scale randomly, but with some overlap!
        loc1 = rng.uniform(-5, 0)
        loc2 = rng.uniform(0, 5)
        scale1 = rng.uniform(1, 3)
        scale2 = rng.uniform(1, 3)

        y = np.concatenate([rng.normal(loc=loc1, scale=scale1, size=tot_n_pts // 2),
                            rng.normal(loc=loc2, scale=scale2, size=tot_n_pts // 2)])
    elif y_distrib == "normal":
        y = rng.normal(size=tot_n_pts)
    elif y_distrib == "lognormal":
        y = rng.lognormal(size=tot_n_pts)
    elif y_distrib == "uniform":
        y = rng.uniform(low=0.0, high=100.0, size=tot_n_pts)
    else:
        raise ValueError(f"Unknown y_distrib: {y_distrib}")
    
    y = torch.tensor(y, dtype=torch.float32, device=device).view(n_pts, n_outs)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    return X, y


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Generate synthetic dataset")
    argparser.add_argument("--n_pts", type=int, default=10_000, help="Number of data points")
    argparser.add_argument("--n_attr", type=int, default=100, help="Number of attributes")
    argparser.add_argument("--n_epochs", type=int, default=5000, help="Number of training epochs")
    argparser.add_argument("--y_distrib", type=str, default="bimodal", help="Distribution of target variable (options: 'bimodal', 'normal', 'lognormal', 'uniform')")
    argparser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    argparser.add_argument("--lmbda", type=float, default=0.15, help="Lambda parameter for loss function")
    argparser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
    argparser.add_argument("--output", type=str, default="dataset.csv", help="Output file name for the generated dataset (CSV format)")
    args = argparser.parse_args()

    n_pts = args.n_pts
    n_attr = args.n_attr
    n_epochs = args.n_epochs
    y_distrib = args.y_distrib

    # TODO: pick these randomly, currently fixed!
    n_cont = 10
    n_ord = 5
    n_categ = [ 10, 3, 3, 3, 5, 1 ]

    assert n_cont + n_ord + sum(n_categ) == n_attr, "n_attr does not match the sum of variable types"

    random_state = args.random_state
    lmbda = args.lmbda
    lr = args.lr
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # NOTE: these for loops are used to 
    # get the results shown in the paper.
    # A single run works for the generation of a single dataset.
    # for random_state in range(40, 50):
    # for lmbda in np.linspace(0, 0.2, 10):


    np.random.seed(random_state)
    torch.manual_seed(random_state)


    X, y = gen_Xy(n_pts, n_attr, y_distrib, device, random_state=random_state)

    model = UnderlyingProcess(n_attr).to(device)
    optimizer = torch.optim.Adam([X], lr=lr)

    # set to 0 to freeze model parameters
    # NOTE: we did observe similar results
    # when using lr_model > 0 (but lr_model << lr,
    # so as to let it change more slowly than X),
    # but for simplicity we set it to 0.
    opt_model = torch.optim.Adam(model.parameters(), lr=0)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        opt_model.zero_grad()
        y_pred = model(X)
        y_pred_lin = get_best_linear_model_predictions(X, y)
        loss = loss_func(y_pred, y_pred_lin, y, lmbda)
        loss.backward()
        optimizer.step()
        opt_model.step()
        if epoch % 250 == 0: # some "logging"
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # sanity check -- will work well for knn, not for LR
    print("LR performance (gold dataset):", get_model_performance(X, y, LinearRegression(), test_size=0.2, random_state=random_state))
    print("KNN performance (gold dataset):", get_model_performance(X, y, KNeighborsRegressor(n_neighbors=5), test_size=0.2, random_state=random_state))

    df = pd.DataFrame(X, columns=[ f"attr_{i}" for i in range(n_attr) ])
    df["target"] = y
    df.to_csv(args.output, index=False)