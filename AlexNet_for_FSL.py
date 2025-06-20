#AlexNet 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Built for FMNIST Dataset
def build_split_model(input_shape=(28, 28, 1), num_classes=10):

    with tf.device('/cpu:0'):
        inputs_client = Input(shape=input_shape)
        # Conv1: 32 filters (original: 96), 5x5, stride=1
        x = Conv2D(32, (5, 5), strides=1, activation='relu', padding='same', name='Conv1')(inputs_client)
        # Conv2: 64 filters (original: 256), 3x3
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='Pool2')(x) #Cut Layer

        client_output = x  
        client_model = models.Model(inputs=inputs_client, outputs=client_output) #Client Side Output


    with tf.device('/gpu:0'):
        # Conv3-5: 64 filters (original: 384/256)
        input_gateway = Input(shape=client_model.output_shape[1:]) #Edge Gateway Side Input
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(input_gateway)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv4')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv5')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='Pool3')(x)
        # Flatten
        x = Flatten(name='Flatten')(x)
        # Dense layers: 512 units (original: 4096)
        x = Dense(512, activation='relu', name='FC1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='FC2')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax', name='Output')(x)
        gateway_model = Model(input_gateway, outputs, name='AlexNet_FashionMNIST_Light') #Edge Gateway Side Output
    return client_model, gateway_model
