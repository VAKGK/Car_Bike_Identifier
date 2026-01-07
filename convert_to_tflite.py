import tensorflow as tf

model = tf.keras.models.load_model("car_bike_cnn_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("car_bike_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
