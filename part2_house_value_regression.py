# import torch
import lhsmdu
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from part1_nn_lib import MultiLayerNetwork, Trainer

class Regressor():

    def __init__(self, x, nb_epoch = 1000, neurons = [16, 16, 16, 1], activations = ["relu", "relu", "relu", "identity"], batchSize = 8, learningRate = 0.1):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 

        net = MultiLayerNetwork(self.input_size, neurons, activations)
        self.trainer = Trainer(
            network=net,
            batch_size=batchSize,
            nb_epoch=nb_epoch,
            learning_rate=learningRate,
            loss_fun="mse",
            shuffle_flag=True,
        )
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_filled = x.fillna(method='ffill').fillna(method='bfill')

        if training:
            self.textual = []
            self.binarizers = []
            for col in x_filled.columns.values:
                if x_filled.dtypes[col] == 'object':
                    self.textual.append(col)
                    lb = preprocessing.LabelBinarizer()
                    lb.fit(x_filled.loc[:, col])
                    self.binarizers.append(lb.classes_)

        for i in range(len(self.textual)):
            lb = preprocessing.LabelBinarizer()
            lb.classes_ = self.binarizers[i]
            one_hot = lb.transform(x_filled.loc[:, self.textual[i]])
            x_filled = x_filled.join(pd.DataFrame(one_hot, columns = self.binarizers[i]))
            x_filled = x_filled.drop(columns = self.textual[i])

        x_np = x_filled.to_numpy()

        if training:
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(x_np)
            self.x_scaler = scaler.scale_, scaler.min_

        scaler = preprocessing.MinMaxScaler()
        scaler.scale_, scaler.min_ = self.x_scaler
        x_np = scaler.transform(x_np)

        if isinstance(y, pd.DataFrame):
            y_np = y.to_numpy()
            if training:
                scaler = preprocessing.MinMaxScaler()
                scaler.fit(y_np)
                self.y_scaler = scaler.scale_, scaler.min_

            scaler = preprocessing.MinMaxScaler()
            scaler.scale_, scaler.min_ = self.y_scaler
            y_np = scaler.transform(y_np)
        
        # Return preprocessed x and y, return None for y if it was None
        return x_np, (y_np if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        self.trainer.train(X ,Y)
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        Y = self.trainer.predict(X)
        scaler = preprocessing.MinMaxScaler()
        scaler.scale_, scaler.min_ = self.y_scaler

        return scaler.inverse_transform(Y)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        _, Y = self._preprocessor(x, y = y, training = False) # Y is normalized target data

        scaler = preprocessing.MinMaxScaler()
        scaler.scale_, scaler.min_ = self.y_scaler
        Y_real = scaler.inverse_transform(Y) # Y_real is real target data

        Y_pre_real = self.predict(x) # Y_pre_real is real (predicted) output data

        return metrics.r2_score(Y_real, Y_pre_real)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def train_validate_test_split(x, y, train_percent, validate_percent, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(x.index)
    m = len(x.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    x_train = x.iloc[perm[:train_end]]
    y_train = y.iloc[perm[:train_end]]
    x_val = x.iloc[perm[train_end:validate_end]]
    y_val = y.iloc[perm[train_end:validate_end]]
    x_test = x.iloc[perm[validate_end:]]
    y_test = y.iloc[perm[validate_end:]]
    return x_train, x_val, x_test, y_train, y_val, y_test


def RegressorHyperParameterSearch(x, y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    results = []

    x_train, x_val, x_test, y_train, y_val, y_test = train_validate_test_split(x, y, 0.6, 0.2, 12)

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    #Batchsize
    #No epochs
    #Learning Rate
    #No of layers
    #No of neurons p layer
    #Activations per layer

    # x, nb_epoch = 1000, neurons = [16, 16, 16, 1], activations = ["relu", "relu", "relu", "identity"], batchSize = 8, learningRate = 0.1)
    # random.choice(neurons), random.choice(activations)

    # Pick a random int between 2 and e.g. 6 --> number of hidden layers:
    #Create empty lists:

    numLayers = 4
    
    for i in range(3, numLayers):
        for _ in range(50):
            minNeurons = 4
            maxNeurons = 30
            sampledNeurons = []
            sampledActivations = []
            samples = np.asarray(lhsmdu.sample((i + 2), 10))

            for j in range(i):
                numNeurons = (samples[j][0] * (maxNeurons - minNeurons)) + minNeurons
                sampledNeurons.append(round(numNeurons))
                if samples[j][1] < 0.5:
                    sampledActivations.append("relu")
                else:
                    sampledActivations.append("sigmoid")

            sampledActivations.append("identity")
            sampledNeurons.append(1)

            # for epochs in (samples[i] * 13) + 3:
            #     for batchSize in (samples[i + 1] * 16) + 2:

            minLearningRate = 0.001
            maxLearningRate = 0.2
            learningRates = (samples[i + 1] * (maxLearningRate - minLearningRate)) + minLearningRate

            for learningRate in learningRates:
                model = Regressor(x, 10, neurons=sampledNeurons, activations = sampledActivations, batchSize=4, learningRate=learningRate)
                model.fit(x_train, y_train)
                model_error = model.score(x_val, y_val)

                print(sampledNeurons, "\t\t", sampledActivations, "\t\t", learningRate, "\t\tR2 score =", model_error)
                results.append([sampledNeurons, sampledActivations, learningRate, model_error])

    return  results# Return the chosen hyper parameters
    # #############################################################################
    # params = {'choice': hp.choice('num_layers',
    #     [ {'layers':'two', },
    #     {'layers':'three', 'neuron3': hp.choice('neuron3', [16,32,64,128,256]),
    #      'activation3':hp.choice('activation3',['sigmoid','relu'])},
    #      {'layers':'four','neuron4': hp.choice('neuron4', [16,32,64,128,256]),
    #      'activation4':hp.choice('activation5',['sigmoid','relu']),
    #      'neuron5': hp.choice('neuron5', [16,32,64,128,256]),
    #     'activation5':hp.choice('activation4',['sigmoid','relu'])}]),
    #     'neuron1': hp.choice('neuron1', [16,32,64,128,256]),
    #     'neuron2': hp.choice('neuron2', [16,32,64,128,256]),
    #     'activation1':hp.choice('activation1',['sigmoid','relu']),
    #     'activation2':hp.choice('activation2',['sigmoid','relu']),
    #     'lr':hp.uniform('lr',1e-4,1e-1),
    #     'batch_size':hp.choice('batch_size',[8,16,32])}
    # grid = GridSearchCV(model, params)
    # start = time.time()
    # grid.fit(trainData, trainLabels)

    # # evaluate the best grid searched model on the testing data
    # print("[INFO] grid search took {:.2f} seconds".format(
    #     time.time() - start))
    # acc = grid.score(testData, testLabels)
    # print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
    # print("[INFO] grid search best parameters: {}".format(
	# grid.best_params_))
    ################################################################################
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    results = RegressorHyperParameterSearch(x_train, y_train)

    df = pd.DataFrame(results)
    df.columns = ['Number of Neurons', 'Activation Functions', 'Learning Rate', 'R2 Score']
    df.to_csv('neural_network_optimisation.csv')
            

    # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

