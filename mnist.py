import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# MNISTデータセットのロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データの前処理
x_train, x_test = x_train / 255.0, x_test / 255.0

# モデルの構築
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
model.fit(x_train, y_train, epochs=5)

# モデルの評価
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# GPUの使用を確認
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
