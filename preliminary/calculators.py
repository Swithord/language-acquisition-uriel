from urielplus.urielplus import URIELPlus
import pandas as pd
import numpy as np


u = URIELPlus()


class DistanceCalculator:
    """
    Base class for calculating language distances for some distance type.
    """
    def __init__(self):
        self.df = None

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate the distance between two languages.
        :param lang1: code for language 1
        :param lang2: code for language 2
        :return: float
        """
        pass

    def _vector_distance(self, lang1: str, lang2: str) -> float:
        """
        Computes language distance by indexing into self.df and calculating the angular distance
        :param lang1: code for language 1
        :param lang2: code for language 2
        :return: float
        """
        if self.df is None:
            raise ValueError("Dataframe is not initialized.")
        if lang1 not in self.df.index:
            print(f"Language {lang1} not found in dataframe.")
            return np.nan
        if lang2 not in self.df.index:
            print(f"Language {lang2} not found in dataframe.")
            return np.nan
        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))
        if not intersection:
            return np.nan
        return u._angular_distance(self.df.loc[lang1].iloc[intersection].to_numpy(),
                                            self.df.loc[lang2].iloc[intersection].to_numpy())


class SyntacticCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("S_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class InventoryCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("INV_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class PhonologicalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("P_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class FeaturalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class MorphologicalCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        df = pd.read_csv(dataset_path, index_col=0)
        self.df = df[[col for col in df.columns if col.startswith("M_")]]

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)


class GenericCalculator(DistanceCalculator):
    def __init__(self, dataset_path: str = None):
        super().__init__()
        if dataset_path is None:
            raise ValueError("Dataset path must be provided.")
        self.df = pd.read_csv(dataset_path, index_col=0)

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        return self._vector_distance(lang1, lang2)
