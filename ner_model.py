import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

import extract_features as ef



def create_model(train=True):
    x_train, y_train, x_test, y_test = ef.extract_features()
    #model bauen
    model = Sequential()
    #input layer
    model.add(Dense(output_dim=405, input_dim=405, activation="relu"))
    #hidden layer
    model.add(Dropout(0.3))
    #model.add(Dense(output_dim=300,input_dim=622,activation="relu"))
    model.add(Dense(output_dim=200, input_dim=405, activation="relu"))
    #output layer
    model.add(Dense(output_dim=9, input_dim=200, activation="softmax"))

    #model compile
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    if train:
        return model, (x_train, y_train)
    else:
        return model, (x_test, y_test)




