import unittest

from part1_nn_lib import *

# linear = LinearLayer(2, 3)
# x = np.ones([4,2])
# linear.forward(x)
# grad_z = np.ones([4,3])
# linear.backward(grad_z)
# learning_rate = 1
# linear.update_params(learning_rate)


class TestPreProcessor(unittest.TestCase):

    def testPreprocessorApplyNormalisationWithOneDimension(self):
        test = np.array([0,1,2,3,4,5,6,7,8,9,10])
        prep_input = Preprocessor(test)
        normalised_test = prep_input.apply(test)
        self.assertEqual(normalised_test.all(), np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).all())

    def testPreprocessorRevertNormalisationWithOneDimension(self):
        test = np.array([0,1,2,3,4,5,6,7,8,9,10])
        prep_input = Preprocessor(test)
        normalised_test = prep_input.apply(test)
        un_normalised_test = prep_input.revert(normalised_test)
        self.assertEqual(un_normalised_test.all(), test.all())

    def testPreprocessorApplyNormalisationWithTwoDimension(self):
        test = np.array([[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10]])
        prep_input = Preprocessor(test)
        normalised_test = prep_input.apply(test)
        self.assertEqual(normalised_test.all(), np.array([[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]).all())

    def testPreprocessorRevertNormalisationWithTwoDimension(self):
        test = np.array([[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10]])
        prep_input = Preprocessor(test)
        normalised_test = prep_input.apply(test)
        un_normalised_test = prep_input.revert(normalised_test)
        self.assertEqual(un_normalised_test.all(), test.all())

if __name__ == '__main__':
    unittest.main()

