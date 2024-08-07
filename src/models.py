from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Activation, add


def residual_module(layer_in, n_filters):
    """Defines a residual module."""
    merge_input = layer_in
    if layer_in.shape[-1] != n_filters:
        merge_input = Conv1D(n_filters, kernel_size=1, padding='same', activation='relu',
                             kernel_initializer='he_normal')(layer_in)
    conv1 = Conv1D(n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(
        layer_in)
    conv2 = Conv1D(n_filters, kernel_size=3, padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    layer_out = add([conv2, merge_input])
    layer_out = Activation('relu')(layer_out)
    return layer_out


def build_inception_model(input_shape, num_classes):
    """Builds the inception model."""
    signal_input = Input(shape=input_shape, name='data')
    layer = residual_module(signal_input, 64)
    layer = Dropout(rate=0.5)(layer)
    layer = residual_module(layer, 64)
    global_avg_pool = GlobalAveragePooling1D(data_format='channels_last')(layer)
    output = Dense(num_classes, activation='softmax', name='predictions')(global_avg_pool)
    model = Model(inputs=signal_input, outputs=output)
    return model
