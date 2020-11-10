import os
import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class DataLoader:
    def __init__(self, dataset: str, verbose: int = 0):
        self._dataset = dataset
        self._verbose = verbose

        self._datasets_folder = 'datasets'

    def load(self):
        if self._dataset == '60':
            return self._load_60()
        elif self._dataset == 'breast_cancer':
            return self._load_bc()
        elif self._dataset == 'chinese_mnist':
            return self._load_cm()
        else:
            return None

    def _load_bc(self):
        filename = 'breast-cancer.csv'
        filepath = os.path.join(self._datasets_folder, filename)

        df = pd.read_csv(filepath)

        X = df.iloc[:,1:]
        y = df.iloc[:,:1]

        dtype_dict: dict = {}
        for col in X.columns:
            dtype_dict[col] = 'category'

        X = X.astype(dtype_dict)

        cat_columns = X.select_dtypes(['category']).columns
        X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

        return X, y

    def _load_60(self):
        X, y = make_classification(n_features=60, n_redundant=20,
                                   n_informative=40, n_samples=100,
                                   n_classes=3, class_sep=0.3,
                                   shuffle=True, random_state=42)

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        return X, y

    def _load_1000(self):
        X, y = make_classification(n_features=1000, n_redundant=700,
                                   n_informative=300, n_samples=200,
                                   n_classes=4, class_sep=0.4,
                                   shuffle=True, random_state=42)

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        return X, y

    def _load_cm(self):
        DATASET = 'chinese_mnist'
        DATA_FILE = 'chinese_mnist.csv'

        DATA_FOLDER = 'data/data'
        FILE_FORMATTER = 'input_{}_{}_{}.jpg'

        DATA_FILE_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FILE)
        DATA_FOLDER_PATH = os.path.join(self._datasets_folder, DATASET, DATA_FOLDER)

        df = pd.read_csv(DATA_FILE_PATH)
        df = df.sample(frac = 1)

        data_size = 100

        X = []
        y = []

        if self._verbose > 0:
            print("Loading data...")

        for i in range(data_size):
            if self._verbose > 0:
                print("Progress:", int(100 * i / data_size), "%", end='\r', flush=True)

            input_data = df.iloc[i]

            value = int(input_data['code'])

            input_image_path = os.path.join(DATA_FOLDER_PATH,
                                            FILE_FORMATTER
                                            .format(input_data['suite_id'],
                                                    input_data['sample_id'],
                                                    value))

            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            input_image = cv2.resize(input_image, (32, 32))

            input_image_vector = input_image.flatten()

            X.append(input_image_vector)
            y.append(value)

        X = pd.DataFrame(np.array(X))
        y = pd.DataFrame(np.array(y))

        if self._verbose > 0:
            print("Done!")

        return X, y
