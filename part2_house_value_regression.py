import torch
import numpy as np

class Regressor():

    def __init__(self, input_size = 9, epoch = 1000, learning_rate = 0.01):
        # Example init parameters for your model: you can remove them or add new ones
        # Remenber to set them with a default value for LabTS tests

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

        X = x # Replace this code with you own
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

        Y = y # Replace this code with you own
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

        X = self._preprocessor(x) # Do not forget to add this line somewhere
        y = x # Replace this code with you own
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
        return 0 # Replace this code with you own


def save_regressor(trained_model): # Alter this function appropriately to work in tandem with load_regressor
    """ Save the trained regressor model in part2_model.pt """

    with open('part2_model.pt', 'wb') as target:
        torch.save(trained_model, target)
    print("\nSaved model in part2_model.pt\n")


def load_regressor(): # Alter this function so that it works in tandem with save_regressor
    """ Load the trained regressor model in part2_model.pt """

    with open('part2_model.pt', 'rb') as target:
        trained_model = torch.load(target)
    print("\nLoaded model in part2_model.pt\n")
    return trained_model



def RegressorHyperParameterSearch(): # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented in the Regressor class.

    Parameters
    ----------

    Returns
    -------        
    The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

