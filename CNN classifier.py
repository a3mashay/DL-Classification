from tensorflow import keras
from tensorflow.keras import backend as K

def mish(x):
        return x * K.tanh(K.softplus(x))

#classification_structure
activation=mish
def gen_model():
    inp=keras.layers.Input([100,4])
    c1=keras.layers.Conv1D(128,3,activation=activation,padding='same')(inp)
    c2=keras.layers.Conv1D(128,3,activation=activation,padding='same')(c1)
    g1=keras.layers.GlobalAveragePooling1D()(c2)
    g2=keras.layers.GlobalMaxPooling1D()(c2)
    out=keras.layers.Dense(1,activation='sigmoid')(keras.layers.BatchNormalization()(keras.layers.add([g1,g2])))
    model=keras.models.Model(inp,out)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model