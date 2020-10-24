import torch
import numpy as np

class Regressor():

    def __init__(self, input_size, hidden_size = 5, epoch = 1000, learning_rate = 0.01):

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return

    def _preprocessor(self, x):
        """ 
        Preprocess input of the network
          
        Parameters
        ----------
        x : raw input of the network

        Returns
        -------        
        X : preprocessed input

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X = x
        return X


    def _preprocessor_output(self, y):
        """ 
        Preprocess output of the network

        Parameters
        ----------
        y : raw ouput of the network

        Returns
        -------        
        Y : preprocessed output

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        Y = y
        return Y

        
    def fit(self, x, y):
        """
        Regressor training function

        Parameters
        ----------
        x : input data
        y : corresponding class vector

        Returns
        -------        
        self : the trained model

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Do not forget to add these lines somewhere
        X = self._preprocessor(x)
        Y = self._preprocessor_output(y)
        return self

            
    def predict(self, x):
        """
        Ouput the value corresponding to an output x.

        Parameters
        ----------
        x : input

        Returns
        -------        
        y : the class x is predicted to belong to

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Do not forget to add this line somewhere
        X = self._preprocessor(x)
        y = x
        return y

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Parameters
        ----------
        x : validation data
        y : validation labels

        Returns
        -------        
        error : cumulated MSE error between the input of the model and teh actual values

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Do not forget to add these lines somewhere
        X = self._preprocessor(x)
        Y = self._preprocessor_output(y)
        return 0


# Please alter this file appropriately to work in tandem with your load_regressor function below
def save_regressor(trained_model):
    """ Save the trained regressor model in part2_model.pt """
    with open('part2_model.pt', 'wb') as target:
        torch.save(trained_model, target)
    print("\nSaved model in part2_model.pt\n")


# Please alter this section so that it works in tandem with the save_regressor method of your class
def load_regressor():
    """ Load the trained regressor model in part2_model.pt """
    with open('part2_model.pt', 'rb') as target:
        trained_model = torch.load(target)
    print("\nLoaded model in part2_model.pt\n")
    return trained_model



# Ensure to add whatever inputs you deem necessary to this function
def RegressorHyperParameterSearch():
    """
    Performs a hyper-parameter for fine-tuning the regressor.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the Regressor class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters

