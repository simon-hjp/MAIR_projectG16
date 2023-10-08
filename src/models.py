####
#     Methods in AI research: part 1b
#     Group 16
####

import numpy as np
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

# our code
from src import classifiers as cl
from src import text_classification as tc


class FeedForwardNeuralNetwork:
    def __init__(self, name="Feed-Forward Neural Network model"):
        self.name = name
        self.oov_token = 0  # Special integer for out-of-vocabulary words
        self.training_length = 0
        self.model = keras.Sequential(
            [
                layers.Dense(240, activation="relu", name="layer1"),
                layers.Dense(120, activation="relu", name="layer2"),
                layers.Dense(60, activation="relu", name="layer3"),
                layers.Dense(30, activation="relu", name="layer4"),
                layers.Dense(15, activation='softmax', name='custom_output_layer')
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    def reshape_x(self, x_data):
        # change the csr_matrix into a regular matrix for our applications, and reshape it at the same time
        self.training_length = x_data.shape[0]
        x_data = [x_data[x, :].toarray().reshape(1, -1)[0] for x in range(self.training_length)]
        x_data = np.array(x_data)
        return x_data

    def fit(self, x_train, y_train):
        # change the csr_matrix into a regular matrix for our applications, and reshape it at the same time
        x_train = self.reshape_x(x_train)

        y_train = [[y_train[x]] for x in range(self.training_length)]
        y_train = np.array(y_train)

        # Number of classes (in our case, 15)
        num_classes = 15

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes=num_classes)

        # Print the training of the model, 0 = nothing printed, 1 = training progress bar, 2 = one line per epoch
        print_training = 2
        history = self.model.fit(x_train, y_train, epochs=7, batch_size=400, validation_split=0.2
                                 , shuffle=True, verbose=print_training)
        # monitoring the training progress and plotting the learning curves
        plot_it = False
        if plot_it:
            # Plot training & validation accuracy values
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()

            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()

    def predict(self, sentences):
        # reshape the data
        sentences = self.reshape_x(sentences)
        # retrieve the model estimate
        prediction = self.model(sentences)
        prediction = np.array(prediction) # was: prediction.numpy()
        # select the most likely answer
        prediction = np.argmax(prediction, axis=1)
        return prediction


def create_models(data_dir):
    """Test each model, report performance, and then initiate the command-line for the user."""

    # Use this boolean to drop duplicate values from the data source
    df_train, df_test = tc.import_data(data_dir=data_dir, drop_duplicates=False)
    df_train_deduplicated, df_test_deduplicated = tc.import_data(data_dir=data_dir, drop_duplicates=True)

    # train the global label encoder and vectorizer
    cl.label_encoder.fit(df_train["dialog_act"])
    cl.vectorizer.fit(df_train["utterance_content"])

    # majority baseline
    majority_model = cl.MajorityBaselineClassifier()
    majority_model.train(df_train)
    tc.evaluate_model(majority_model, df_test)

    # rule-based baseline
    rule_model = cl.RuleBaselineClassifier()
    tc.evaluate_model(rule_model, df_test)

    # decision tree
    dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }

    dt_classifier = cl.DialogActsClassifier(name="Decision tree classifier")
    dt_classifier.train(df_train["utterance_content"], df_train["dialog_act"], fit_hyperparams=True)
    tc.evaluate_model(dt_classifier, df_test)

    # decision tree without duplicates
    dt_classifier_dd = cl.DialogActsClassifier(name="Decision tree without duplicates")
    dt_classifier_dd.train(df_train_deduplicated["utterance_content"],
                           df_train_deduplicated["dialog_act"],
                           fit_hyperparams=True
                           )
    tc.evaluate_model(dt_classifier_dd, df_test_deduplicated)

    # logistic regression
    lr_classifier = cl.DialogActsClassifier(model=cl.LogisticRegression(max_iter=10000), name="Logistic regression")
    lr_classifier.train(df_train["utterance_content"], df_train["dialog_act"])
    tc.evaluate_model(lr_classifier, df_test)

    # logistic regression without duplicates
    lr_classifier_dd = cl.DialogActsClassifier(model=cl.LogisticRegression(max_iter=10000),
                                               name="Logistic regression without duplicates")
    lr_classifier_dd.train(df_train_deduplicated["utterance_content"], df_train_deduplicated["dialog_act"])
    tc.evaluate_model(lr_classifier_dd, df_test_deduplicated)
    
    # Feed-Forward Neural Network
    ffnn_classifier = cl.FeedForwardNeuralNetworkClassifier(name="Feed-Forward Neural Network")
    ffnn_classifier.train(df_train["utterance_content"], df_train["dialog_act"])
    tc.evaluate_model(ffnn_classifier, df_test)

    # Feed-Forward Neural Network without duplicates
    ffnn_classifier_dd = cl.FeedForwardNeuralNetworkClassifier(name="Feed-Forward Neural Network without duplicates")
    ffnn_classifier_dd.train(df_train_deduplicated["utterance_content"], df_train_deduplicated["dialog_act"])
    tc.evaluate_model(ffnn_classifier_dd, df_test_deduplicated)

    should_pickle = False
    # Pickle the object and save it to a file
    if should_pickle:
        file_path = 'FeedForwardsNeuralNetwork-deDuplicated.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(ffnn_classifier_dd, file)


    # add all models to a dictionary and return it
    models_dict = {
        'A': majority_model,
        'B': rule_model,
        'C': dt_classifier,
        'D': dt_classifier_dd,
        'E': lr_classifier,
        'F': lr_classifier_dd,
        'G': ffnn_classifier,
        'H': ffnn_classifier_dd
    }

    return models_dict
