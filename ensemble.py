import os
import pandas as pd

if __name__ == "__main__":
    ensemble = pd.read_csv(os.path.join(
        os.getcwd(), 'DATA', 'sample_submission.csv'))
    model1 = pd.read_csv(os.path.join(
        os.getcwd(), 'submissions', 'Model1.csv'))
    model2 = pd.read_csv(os.path.join(
        os.getcwd(), 'submissions', 'Model2.csv'))
    ensemble.iloc[:, 1:] = (0.3*model1.iloc[:, 1:] +
                            0.7*model2.iloc[:, 1:]) / 2
    ensemble.to_csv('/content/ensemble_submission.csv', index=False)
