import tensorflow as tf

# If you want to use tensorflow with GPU computations you have to install compatible version of CUDA and cuDNN
# for tensorflow 2.11 ytou need CUDA 11.2 and cuDNN 8.1
# https://www.tensorflow.org/install/source#gpu
# HOW TO INSTALL CUDA IN WINDOWS 10
# https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255



if __name__ == '__main__':
    #gpu_available = tf.test.is_gpu_available()
    #is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    #is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3, 0))
    #tf.config.list_physical_devices('GPU')
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)

    import tensorflow as tf
    from tensorflow.python.client import device_lib

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    device_lib.list_local_devices()