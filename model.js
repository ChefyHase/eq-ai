const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model = null;
  }

  build() {
    const droprate = 0;
    const input = tf.input({ shape: [4096] });
    const reshape = tf.layers.reshape({ targetShape: [4096, 1, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({
      filters: 128,
      kernelSize: [64, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(reshape);
    const norm1 = tf.layers.batchNormalization().apply(conv1);
    const conv2 = tf.layers.conv2d({
      filters: 128,
      kernelSize: [32, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(norm1);
    const norm2 = tf.layers.batchNormalization().apply(conv2);
    const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm2);
    const dropout1 = tf.layers.dropout({ rate: droprate }).apply(pool1);

    const conv3 = tf.layers.conv2d({
      filters: 256,
      kernelSize: [16, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(dropout1);
    const norm3 = tf.layers.batchNormalization().apply(conv3);
    const conv4 = tf.layers.conv2d({
      filters: 256,
      kernelSize: [8, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(norm3);
    const norm4 = tf.layers.batchNormalization().apply(conv4);
    const pool2 = tf.layers.globalMaxPooling2d({ name: 'pool' }).apply(norm4);
    const dropout2 = tf.layers.dropout({ rate: droprate }).apply(pool2);

    const dense1 = tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropout2);
    const dense1norm = tf.layers.batchNormalization().apply(dense1);
    const dense2 = tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dense1norm);
    const dense2norm = tf.layers.batchNormalization().apply(dense2);
    const dense3 = tf.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dense2norm);
    const dense3norm = tf.layers.batchNormalization().apply(dense3);
    const denseOutput = tf.layers.dense({
      units: 3,
      activation: 'softmax',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dense3norm);

    const model = tf.model({ inputs: input, outputs: denseOutput });
    model.summary();
    this.model = model;

    return model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.005);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError', metrics: ['accuracy'] });

    for (let i = 0; i < config.trainEpoches; ++i) {
      let { xs, ys } = this.data.nextBatch();

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
    // const model = await tf.loadLayersModel(`file://${__dirname}/mastering-ai/model.json`);
    if (this.model === null) this.build();
    const predictBatch = await this.data.predictBatch(wav);
    const prediction = this.model.predict(predictBatch);
    const mean = prediction.mean(0).dataSync();
    const freq = this.data.invNorm(mean[0], 20, 20000);
    const gain = this.data.invNorm(mean[1], -15.0, 15.0);
    const q = this.data.invNorm(mean[2], 0.1, 24.0);

    return { freq, gain, q };
  }
}

module.exports = new Model();
