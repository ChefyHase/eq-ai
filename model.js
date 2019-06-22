const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model;
  }

  build() {
    const droprate = 0.5;
    const input = tf.input({ shape: [2048] });
    const reshape = tf.layers.reshape({ targetShape: [64, 32, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({ filters: 16, kernelSize: 32, strides: 1, padding: 'same' }).apply(reshape);
    const activ1 = tf.layers.activation('relu').apply(conv1);
    const norm1 = tf.layers.batchNormalization().apply(activ1);
    const pooling1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(norm1);
    const dropout1 = tf.layers.dropout({ rate: droprate }).apply(pooling1);

    const conv2 = tf.layers.conv2d({ filters: 32, kernelSize: 5, strides: 1, padding: 'same' }).apply(dropout1);
    const activ2 = tf.layers.activation('relu').apply(conv2);
    const norm2 = tf.layers.batchNormalization().apply(activ2);
    const pooling2 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(norm2);
    const dropout2 = tf.layers.dropout({ rate: droprate }).apply(pooling2);

    const flatten = tf.layers.flatten().apply(dropout2);

    const dense1 = tf.layers.dense({ units: 1000 }).apply(flatten);
    const activDense1 = tf.layers.activation('relu').apply(dense1);

    const dense2 = tf.layers.dense({ units: 500 }).apply(activDense1);
    const activDense2 = tf.layers.activation('relu').apply(dense2);

    const dense3 = tf.layers.dense({ units: 100 }).apply(activDense2);
    const activDense3 = tf.layers.activation('relu').apply(dense3);

    const denseOutput = tf.layers.dense({ units: 3 }).apply(activDense3);
    const activOutput = tf.layers.activation('relu').apply(denseOutput);

    const model = tf.model({ inputs: input, outputs: activOutput });
    model.summary();
    this.model = model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0001);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError' });

    for (let i = 0; i < config.trainEpoches; ++i) {
      const { xs, ys } = this.data.nextBatch();

      const h = await this.model.fit(xs, ys, {
        batchSize: 200,
        epochs: 100,
        shuffle: true,
        validationSplit: 0.3
      });

      console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
      await this.model.save('file://./mastering-ai');

      xs.dispose();
      ys.dispose();
    }
  }

  async predict(wav) {
    const model = await tf.loadLayersModel(`file://${__dirname}/mastering-ai/model.json`);
    const predictBatch = await this.data.predictBatch(wav);
    const prediction = model.predict(predictBatch);
    prediction.print()
    const mean = prediction.mean(0).dataSync();
    const freq = this.data.invNorm(mean[0], 20, 20000);
    const gain = this.data.invNorm(mean[1], -15.0, 15.0);
    const q = this.data.invNorm(mean[2], 0.1, 24.0);

    return { freq, gain, q };
  }
}

module.exports = new Model();
