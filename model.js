const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model;
  }

  build() {
    const droprate = 0.1;
    const input = tf.input({ shape: [2048] });
    const reshape = tf.layers.reshape({ targetShape: [2048, 1, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({ filters: 64, kernelSize: [1024, 1], strides: 8, padding: 'same' }).apply(reshape);
    const activ1 = tf.layers.leakyReLU().apply(conv1);
    const norm1 = tf.layers.batchNormalization().apply(activ1);
    const pooling1 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm1);
    const dropout1 = tf.layers.dropout({ rate: droprate }).apply(pooling1);

    const conv2 = tf.layers.conv2d({ filters: 32, kernelSize: [512, 1], strides: 1, padding: 'same' }).apply(dropout1);
    const activ2 = tf.layers.leakyReLU().apply(conv2);
    const norm2 = tf.layers.batchNormalization().apply(activ2);
    const pooling2 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm2);
    const dropout2 = tf.layers.dropout({ rate: droprate }).apply(pooling2);

    const conv3 = tf.layers.conv2d({ filters: 8, kernelSize: [512, 1], strides: 1, padding: 'same' }).apply(dropout2);
    const activ3 = tf.layers.leakyReLU().apply(conv3);
    const norm3 = tf.layers.batchNormalization().apply(activ3);
    const pooling3 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm3);
    const dropout3 = tf.layers.dropout({ rate: droprate }).apply(pooling3);

    const conv4 = tf.layers.conv2d({ filters: 2, kernelSize: [512, 1], strides: 1, padding: 'same' }).apply(dropout3);
    const activ4 = tf.layers.leakyReLU().apply(conv4);
    const norm4 = tf.layers.batchNormalization().apply(activ4);
    const pooling4 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm4);
    const dropout4 = tf.layers.dropout({ rate: droprate }).apply(pooling4);

    const flatten = tf.layers.flatten().apply(dropout4);

    const dense1 = tf.layers.dense({ units: 25 }).apply(flatten);
    const activDense1 = tf.layers.leakyReLU().apply(dense1);

    const denseOutput = tf.layers.dense({ units: 4 }).apply(activDense1);
    const activOutput = tf.layers.leakyReLU().apply(denseOutput);

    const model = tf.model({ inputs: input, outputs: activOutput });
    model.summary();
    this.model = model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0001);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError', metrics: ['accuracy'] });

    for (let i = 0; i < config.trainEpoches; ++i) {
      const { xs, ys } = this.data.nextBatch();

      const h = await this.model.fit(xs, ys, {
        batchSize: 500,
        epochs: 500,
        shuffle: true,
        validationSplit: 0.4
      });

      console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
      await this.model.save('file://mastering-ai');

      xs.dispose();
      ys.dispose();

      // let buffer = await data.separateSound(__dirname + '/test.wav');
      // let xs = data.fft(tf.tensor(buffer));
    }
  }
}

module.exports = new Model();
