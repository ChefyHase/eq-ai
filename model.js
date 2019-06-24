const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model;
  }

  build() {
    const droprate = 0;
    const input = tf.input({ shape: [2048] });
    const reshape = tf.layers.reshape({ targetShape: [32, 32, 2] }).apply(input);

    const conv1 = tf.layers.conv2d({
      filters: 16,
      kernelSize: [3, 1],
      padding: 'same',
      activation: 'linear',
      kernelInitializer: 'heNormal'
    }).apply(reshape);
    const norm1 = tf.layers.batchNormalization().apply(conv1);
    const pool1 = tf.layers.averagePooling2d({ poolSize: [2, 1] }).apply(norm1);
    const dropout1 = tf.layers.dropout({ rate: droprate }).apply(pool1);

    const conv2 = tf.layers.conv2d({
      filters: 32,
      kernelSize: [3, 1],
      padding: 'same',
      activation: 'linear',
      kernelInitializer: 'heNormal'
     }).apply(dropout1);
    const norm2 = tf.layers.batchNormalization().apply(conv2);
    const pool2 = tf.layers.averagePooling2d({ poolSize: [2, 1] }).apply(norm2);
    const dropout2 = tf.layers.dropout({ rate: droprate }).apply(pool2);

    const conv3 = tf.layers.conv2d({
      filters: 64,
      kernelSize: [3, 1],
      padding: 'same',
      activation: 'linear',
      kernelInitializer: 'heNormal'
     }).apply(dropout2);
    const norm3 = tf.layers.batchNormalization().apply(conv3);
    const pool3 = tf.layers.averagePooling2d({ poolSize: [2, 1] }).apply(norm3);
    const dropout3 = tf.layers.dropout({ rate: droprate }).apply(pool3);

    const flatten = tf.layers.flatten().apply(dropout3);

    const dense1 = tf.layers.dense({
      units: 5000,
      activation: 'sigmoid',
      kernelInitializer: 'heNormal'
     }).apply(flatten);
    const dense1norm = tf.layers.batchNormalization().apply(dense1);
    const dense2 = tf.layers.dense({
      units: 1024,
      activation: 'sigmoid',
      kernelInitializer: 'heNormal'
     }).apply(dense1norm);
    const dense2norm = tf.layers.batchNormalization().apply(dense2);
    const denseOutput = tf.layers.dense({
      units: 3,
      activation: 'sigmoid',
      kernelInitializer: 'heNormal'
     }).apply(dense2norm);

    const model = tf.model({ inputs: input, outputs: denseOutput });
    model.summary();
    this.model = model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0005);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError', metrics: ['accuracy'] });

    for (let i = 0; i < config.trainEpoches; ++i) {
      const { xs, ys } = this.data.nextBatch();

      const h = await this.model.fit(xs, ys, {
        batchSize: 64,
        epochs: 100,
        shuffle: true,
        validationSplit: 0.3,
        // callbacks: tf.callbacks.earlyStopping()
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
