from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

def add_mlp(model, layers):
    for i, nodes in enumerate(layers):
        activation = 'relu' if i < len(layers) - 1 else 'softmax'
        model.add(Dense(nodes, activation=activation, name=f'fc{i + 1}'))
    return model


INPUT = (784)
MLP = (40,20,10)

model = Sequential([
    InputLayer(input_shape=INPUT),])

model = add_mlp(model, MLP)

model.summary()
