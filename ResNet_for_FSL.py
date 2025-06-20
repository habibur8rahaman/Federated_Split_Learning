#ResNet18 (cut layer: 3rd layer)

def build_split_model(input_shape=(28, 28, 1), num_classes=10):
    with tf.device('/cpu:0'):
        inputs_client = layers.Input(shape=input_shape)
        x = layers.Rescaling(1./255)(inputs_client)
        
        # Initial conv (smaller kernel, no stride)
        x = layers.Conv2D(32, (3, 3), strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        client_output = layers.ReLU()(x) #Cut Layer
        client_model = models.Model(inputs=inputs_client, outputs=client_output) #Client Side Output

    with tf.device('/gpu:0'):
        # Residual blocks (3 stages)
        def residual_block(x, filters, downsample=False):
            shortcut = x
            stride = 2 if downsample else 1
            x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            if downsample:
                shortcut = layers.Conv2D(filters, (1, 1), strides=2, use_bias=False)(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            return x

        input_gateway = layers.Input(shape=client_model.output.shape[1:]) #Edge Gateway Side Input
        # Stage 1-3
        x = residual_block(input_gateway, 32)  # Stage 1
        x = residual_block(x, 32)
        x = residual_block(x, 64, downsample=True)  # Stage 2
        x = residual_block(x, 64)
        x = residual_block(x, 128, downsample=True)  # Stage 3
        x = residual_block(x, 128)
        
        # Final layers
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        gateway_model = models.Model(inputs=input_gateway, outputs=outputs) #Edge Gateway Side Output
    return client_model, gateway_model
