import unittest

import pandas as pd
import numpy as np

from src.alg_eda import mypca, mykmeans

class TestMyPCA(unittest.TestCase):
    def setUp(self):
        self.mypca = mypca(n_components=2)
        self.df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

    def test_fit(self):
        result = self.mypca.fit(self.df)
        self.assertIsInstance(result, list)  
        if result: 
            self.assertIsInstance(result[0], tuple)  
            self.assertEqual(len(result[0]), 2)  
            self.assertTrue(all(isinstance(i, str) for i in result[0])) 
            
            
class TestMyKMeans(unittest.TestCase):
    def setUp(self):
        self.mykmeans = mykmeans(n_clusters=2)
        self.df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

    def test_fit(self):
        result = self.mykmeans.fit(self.df)
        self.assertIsInstance(result, np.ndarray)  # проверка, что результат является массивом numpy
        self.assertEqual(len(result), len(self.df))  # проверка, что длина массива совпадает с количеством строк в df

    def test_get_silhouette_score(self):
        self.mykmeans.fit(self.df)
        score = self.mykmeans.get_silhouette_score()
        self.assertIsInstance(score, float)  # проверка, что результат является числом с плавающей точкой


if __name__ == '__main__':
    unittest.main()
