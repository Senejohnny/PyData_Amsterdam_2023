""" 
Model Training End2End pipeline 
Author: Danial Senejohnny 
"""
import warnings
import pandas as pd
import numpy as np
from pandas.core.common import SettingWithCopyWarning
import tensorflow as tf
import keras
from keras.initializers import Constant
from keras import  layers
from gensim.models import Word2Vec

from src.model_build.common.model_utility import (
    CustomScaler,
    get_embedding_weights,
    df_train_test_split
)


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

path_embedding = "~/Repos/STRIDE/survival-models/local/ep_skipgram.model"
embedding = Word2Vec.load(path_embedding)

maxlen = 60

df = (
    data_loading(path_data)
)

train_data, test_data = df_train_test_split(df, donor_type , test_size=0.25)

#########################################
# Keras Model
#########################################
vectorizer = layers.TextVectorization(
                            standardize=None,
                            output_mode='int',
                            vocabulary=all_epitopes,
                            output_sequence_length=maxlen,
                            name='Vectorization_Layer')

embedding_matrix = get_embedding_weights(embedding.wv, vectorizer)
vocab = vectorizer.get_vocabulary(include_special_tokens=False)
vocab_size = len(vocab) + 2 #Account for textvectorization padding token ('') and Out Of Vocab token ('[UNK]')

##%%%%%%%%%%%%%%%%%%%%%%
features_shape = train_data[0].shape[1] - 1 # num of features excluding the epitope column
embedding_dim = embedding.wv.vector_size
input_features = layers.Input(shape=[features_shape], name="input_features")
input_epitopes_string = layers.Input(shape=(1,), dtype=tf.string, name="input_epitopes")
epitopes_vector = vectorizer(input_epitopes_string)
epitope_emb = layers.Embedding(
                            input_dim=vocab_size, 
                            output_dim=embedding_dim,
                            input_length=maxlen,
                            embeddings_initializer=Constant(embedding_matrix),
                            trainable=False,
                            name="Embedding_Layer")(epitopes_vector) 
# To concat with structured features embedding output is matrix and needs to be flatten to vector
flattened_epitope_emb = layers.Flatten()(epitope_emb)
concat = layers.concatenate([input_features, flattened_epitope_emb])
hidden1 = layers.Dense(3, activation='relu', name="Dense_Layer_1")(concat)
hidden2 = layers.Dense(5, activation='relu', name="Dense_Layer_2")(hidden1)
output = layers.Dense(1, activation='linear')(hidden2) # "linear" activation: a(x) = x, i.e. no activation is applied
##%%%%%%%%%%%%%%%%%%%%%%
model = keras.Model(
    inputs=[input_features, input_epitopes_string], 
    outputs=output
)

#########################################
# Model Training
#########################################
from tf_survival_ep import DeepSurvival

deepsurv = DeepSurvival(model=model, learning_rate=0.005, epochs=15)
deepsurv.train_evaluate(
    train_dataset=train_data, 
    val_dataset=test_data, 
    batch_size=32
)

X, *_ = train_data
vals = slice(10,16)
# ============================
# X = tf.convert_to_tensor(X.values[0:5][..., np.newaxis])
# X = tf.data.Dataset.from_tensor_slices(X.values[0:5]) 
# print(deepsurv.predict(X.iloc[10:12]))
# X.iloc[10:12]  
# print(surv_funcs := deepsurv.predict_survival_function(X.iloc[10:16]))
# ============================
deepsurv.plot_survival_function(X.iloc[vals])