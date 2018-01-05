import unittest
from torch import FloatTensor, LongTensor
from evaluation import f1_score as score


class Test_f1_score(unittest.TestCase):

    def test_f1_macro(self):
        target = LongTensor([[1, 1, 1, 0, 1], [1, 0, 1, 0, 1]])
        pred = FloatTensor([[1, -1, 1, -1, 1], [-1, 1, 1, -1, -1]])
        f1_macro = score.f1_macro(target, pred, 5, 0.5)
        self.assertAlmostEqual(0.466667, f1_macro, 5)

    def test_f1_score(self):
        target = LongTensor([[1, 1, 1, 0, 1], [1, 0, 1, 0, 1]])
        pred = FloatTensor([[1, -1, 1, -1, 1], [-1, 1, 1, -1, -1]])
        f1_macro, f1_micro = score.f1_score(target, pred, 5, threshold=0.5)
        self.assertAlmostEqual(7 / 15, f1_macro, 5)
        self.assertAlmostEqual(2 / 3, f1_micro, 5)

        pred = FloatTensor([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0]])
        f1_macro, f1_micro = score.f1_score(
            target, pred, 5, use_threshold=False)
        self.assertAlmostEqual(7 / 15, f1_macro, 5)
        self.assertAlmostEqual(2 / 3, f1_micro, 5)
