#LeNet (cut layer: 2nd layer)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense

#Built for CIFAR-10 Dataset
def build_split_model(input_shape=(32, 32, 3), num_classes=10):

    with tf.device('/cpu:0'):
        inputs_client = Input(shape=input_shape)
        # C1: 6 filters, 5x5, tanh
        x = Conv2D(6, (5, 5), activation='tanh', padding='valid', name='C1')(inputs_client)
        # S2: Average pooling 2x2, stride 2
        client_output = AveragePooling2D(pool_size=(2, 2), name='S2')(x) #Cut Layer
        client_model = models.Model(inputs_client, outputs= client_output) #Client Side Output


    with tf.device('/gpu:0'):
        inputs_gateway = Input(shape = client_model.output.shape[1:]) #Edge Gateway Side Input
        # C3: 16 filters, 5x5, tanh
        x = Conv2D(16, (5, 5), activation='tanh', padding='valid', name='C3')(inputs_gateway)
        # S4: Average pooling 2x2, stride 2
        x = AveragePooling2D(pool_size=(2, 2), name='S4')(x)
        # Flatten
        x = Flatten(name='Flatten')(x)
        # F5: Dense 120, tanh
        x = Dense(120, activation='tanh', name='F5')(x)
        # F6: Dense 84, tanh
        x = Dense(84, activation='tanh', name='F6')(x)
        # Output: Softmax for 10 classes
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
        gateway_model = Model(inputs_gateway, outputs, name='LeNet-5_CIFAR10') #Edge Gateway Side Output


    return client_model, gateway_model
