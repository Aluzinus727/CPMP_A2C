import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD


def create_critic_model():
    state_input = Input(shape=(6, 7))
    state_h1 = Dense(100, activation='relu')(state_input)
    state_h2 = Dense(100, activation='relu')(state_h1)
    state_h3 = Dense(150, activation='relu')(state_h2)
    output = Flatten()(state_h3)
    output = Dense(1, activation=None)(output)

    model = Model(inputs=state_input, outputs=[output])

    adam  = Adam(learning_rate=0.001)
    model.compile(loss="mae", optimizer=adam)

    return model

def create_actor_model():
    n_actions = 20

    state_input = Input(shape=(6, 7))
    h1 = Dense(100, activation='relu')(state_input)
    h2 = Dense(250, activation='relu')(h1)
    h3 = Dense(50, activation='relu')(h2)
    output = Flatten()(h3)
    output = Dense(n_actions, activation="softmax")(output)

    model = Model(state_input, outputs=[output])
    adam  = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model