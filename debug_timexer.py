import pandas as pd
import numpy as np
import torch
from preprocessing.walk_forward import wfrw
from model.bases_models.timexer_model import objective_timexer_global
import optuna

# Create dummy train data
N = 1000
dt = pd.date_range('2020-01-01', periods=N)
X = pd.DataFrame(np.random.rand(N, 14), index=dt)
y = pd.Series(np.random.rand(N), index=dt)

def test_optuna():
    sp = wfrw(y, k=2, fh_val=30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing TimeXer trial...")
    
    def obj(trial):
        return objective_timexer_global(
            trial, X, y, sp, 
            device=device,
            seq_len=96, pred_len=30, features='MS'
        )
    
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=1)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

if __name__ == "__main__":
    test_optuna()
