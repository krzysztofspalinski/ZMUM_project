import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPool1D, Bidirectional, LSTM, Embedding, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K

class model_1:

	

	def __init__(self):
		pass


	def get_model(embedding_matrix, maxlen, max_features, embed_size, len_features):

		NUM_CLASSES = 10
		INPUT_SHAPE = (32, 32, 3)

		inp = Input(shape=(maxlen,))
		x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
		x = Bidirectional(LSTM(128, return_sequences=True))(x)
		x = GlobalMaxPool1D()(x)
		x = Dense(64, activation="relu")(x)

		inp_meta = Input(shape=(len_features,))
		y = Dense(16, activation='relu')(inp_meta)
		concat = Concatenate()([x, y])
		concat = Dense(32, activation="relu")(concat)
		output = Dense(3, activation="softmax")(concat)

		model = Model(inputs=[inp, inp_meta], outputs=output)

		return model


	def describe():
		tmp_model = model_1.get_model()
		tmp_model.summary()

