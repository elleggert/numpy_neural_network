import torch
import pickle
import numpy as np
import pandas as pd

class Regressor():

    def __init__(self, input_size, epoch = 1000):
        # You can add any input parameters you need
        # Remenber to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            input_size {int} -- input size of the model.
            epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.input_size = input_size 
        self.epoch = epoch 
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x):
        """ 
        Preprocess input of the network.
          
        Arguments:
            x {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw input array of shape (batch_size, input_size).

        Returns:
            X {torch.tensor} 
                -- Preprocessed input array of size (batch_size, input_size).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X = x # Replace this code with your own
        return X

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def _preprocessor_output(self, y):
        """ 
        Preprocess output of the network.

        Arguments:
            y {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw ouput array of shape (batch_size, 1).

        Returns:
            Y {torch.tensor} 
                -- Preprocessed output array of size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        Y = y # Replace this code with your own
        return Y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            x {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw input array of shape (batch_size, input_size).
            y {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Do not forget to add these lines somewhere
        X = self._preprocessor(x)
        Y = self._preprocessor_output(y)
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an output x.

        Arguments:
            x {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw input array of shape (batch_size, input_size).

        Returns:
            y {np.darray} 
                -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X = self._preprocessor(x) # Do not forget to add this line somewhere
        y = x # Replace this code with your own
        return y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            x {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw input array of shape (batch_size, input_size).
            y {np.ndarray or pd.DataFrame or torch.tensor} 
                -- Raw ouput array of shape (batch_size, 1).

        Returns:
            error {float} 
                -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Do not forget to add these lines somewhere
        X = self._preprocessor(x)
        Y = self._preprocessor_output(y)
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    # Alter this function appropriately to work in tandem with load_regressor
    """ Save the trained regressor model in part2_model.pickle """

    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    # Alter this function so that it works in tandem with save_regressor
    """ Load the trained regressor model in part2_model.pickle """

    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor 
    implemented in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data 
    # as it contains various object types (numerical, text)
    # Feel free to use another CSV reader tool
    data = pd.read_csv("housing.csv") 

    # Split train-val sets
    split_idx = int(0.8 * len(data))
    x_train = data.loc[:, data.columns != output_label].iloc[:split_idx]
    y_train = data[output_label].iloc[:split_idx]
    x_val = data.loc[:, data.columns != output_label].iloc[split_idx:]
    y_val = data[output_label].iloc[split_idx:]

    # Training
    regressor = Regressor(len(x_train.columns), epoch = 1000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_val, y_val)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

