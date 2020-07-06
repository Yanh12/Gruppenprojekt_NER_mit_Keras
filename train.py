import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

import extract_features as ef
import eval_and_predict as ep

""""" bit die ausklammernde Linien wieder herstellen, wenn ihr das Model erneuert trainiert 
import ner_model
model, (train_x, train_y) = ner_model.create_model()
# train model
model.fit(train_x, train_y,batch_size=50,epochs=30)
model.save('model_weights.h5')

