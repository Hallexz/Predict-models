import sys
sys.path.insert(0, 'Predict-students/src/eda.py')

import unittest

import pandas as pd
import numpy as np
import os

from eda import process_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("/home/hallex/spyd/study/Predict-students/tests/data/nasa.csv")

    def test_remove_duplicates(self):
        df = process_data(self.df)
        self.assertTrue(df.duplicated().sum() == 0, "Обработанный dataframe имеет дубликаты")

    def test_replace_nan(self):
        df = process_data(self.df)
        self.assertTrue(df.isnull().sum().sum() == 0, "Обработанный dataframe имеет значения NaN")

    def test_numeric_conversion(self):
        df = process_data(self.df)
        self.assertTrue((df.dtypes == 'float64').all(), "Обработанный dataframe имеет нечисловые значения")


if __name__ == '__main__':
    unittest.main()
