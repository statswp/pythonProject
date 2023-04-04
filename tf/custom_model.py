import tensorflow as tf


class CustomModel(tf.keras.Model):
    def __init__(self, ):
        super(CustomModel, self).__init__()
        self.dense1 = None
        self.dense2 = None
        self.bn = None

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(32)
        self.dense2 = tf.keras.layers.Dense(1)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn(x)
        out = self.dense2(x)
        return out


if __name__ == '__main__':
    import pandas as pd

    # 数据准备
    df = pd.read_csv('/Users/wp/PycharmProjects/pythonProject/data/criteo_sample.txt')
    cols = ["label"] + [c for c in df.columns if c.startswith("I")]
    inputs = df[cols].fillna(0)

    label = inputs.pop("label")
    dataset = tf.data.Dataset.from_tensor_slices((inputs, label)).shuffle(128).batch(16)
    # for f, l in dataset:
    #     print(f.shape)
    #     print(l.shape)
    #     print(f)
    #     print(l)
    #     break

    # 训练模型
    model = CustomModel()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(dataset, verbose=1, epochs=10)
    print(history.history)
