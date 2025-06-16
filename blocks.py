from keras.layers import Conv2D, UpSampling2D, Concatenate, Add, BatchNormalization, Activation, Multiply

# batch normalization and activation function
def batch_norm_act(x, act=True):
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x

# convolutional block with batch normalization and activation
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = batch_norm_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

# residual block with two convolutional blocks and a shortcut connection
def residual_block(x, filters, strides=1):
    res = conv_block(x, filters, strides=strides)
    res = conv_block(res, filters, strides=1)
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)
    shortcut = batch_norm_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

# attention gate
def attention_gate(Fg, Fs, filters):
    Wg = Conv2D(filters, kernel_size=1, padding="same")(Fg)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(filters, kernel_size=1, padding="same")(Fs)
    Ws = BatchNormalization()(Ws)

    psi = Activation("relu")(Add()([Wg, Ws]))
    psi = Conv2D(1, kernel_size=1, padding="same")(psi)
    psi = Activation("sigmoid")(psi)

    return Multiply()([Fs, psi])

# upsample concat block that upsamples the input and concatenates it with a skip connection
def upsample_concat_block(x, xskip):
    up = UpSampling2D((2, 2))(x)
    con = Concatenate()([up, xskip])
    return con
