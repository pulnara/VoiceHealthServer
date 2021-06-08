# skrypt minimalny do rozpoznania próbki
import tensorflow as tf

def predict(audio_binary):

    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    audio, _ = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(audio, axis=-1)
    zero_padding = tf.zeros([500000] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)

    interpreter.set_tensor(input_index, [spectrogram])
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    prediction = tf.nn.softmax(prediction[0])

    print(prediction)

    return tf.keras.backend.get_value(prediction[0]), tf.keras.backend.get_value(prediction[1])