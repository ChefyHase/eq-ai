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
    const input = tf.input({ shape: [512] });
    const reshape = tf.layers.reshape({ targetShape: [512, 1, 1] }).apply(input);

    const outputs = [];
    for (let n = 0; n < 3 * 8; n++) {

      const conv1 = tf.layers.conv2d({
        filters: 32,
        kernelSize: [128, 1],
        padding: 'same',
        activation: 'relu',
        // kernelRegularizer: 'l1l2',
        strides: [2, 1],
        kernelInitializer: 'heNormal'
      }).apply(reshape);
      const conv2 = tf.layers.conv2d({
        filters: 32,
        kernelSize: [128, 1],
        padding: 'same',
        activation: 'relu',
        // kernelRegularizer: 'l1l2',
        strides: [2, 1],
        kernelInitializer: 'heNormal'
      }).apply(conv1);
      const norm1 = tf.layers.batchNormalization().apply(conv2);
      const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm1);

      const flatten = tf.layers.flatten().apply(pool1)

      const dense1 = tf.layers.dense({
        units: 256,
        activation: 'relu',
        // kernelRegularizer: 'l1l2',
        kernelInitializer: 'heNormal'
      }).apply(flatten);
      const dense2 = tf.layers.dense({
        units: 128,
        activation: 'relu',
        // kernelRegularizer: 'l1l2',
        kernelInitializer: 'heNormal'
      }).apply(dense1);
      const denseOutput = tf.layers.dense({
        units: 2,
        activation: 'softmax',
        // kernelRegularizer: 'l1l2',
        kernelInitializer: 'heNormal'
      }).apply(dense2);

      outputs.push(denseOutput);
    }

    const model = tf.model({ inputs: input, outputs: outputs });
    model.summary();
    this.model = model;

    return model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0001);
    this.model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    for (let j = 0; j < config.trainEpoches; ++j) {
      let { xs, ys } = this.data.nextBatch();
      for (let i = 0; i < 50; i++) {
        const h = await this.model.fit(xs, ys, {
          batchSize: 20,
          epochs: 1,
          shuffle: true,
          validationSplit: 0.3,
          verbose: 0
        });

        console.log("Epoch " + i + " : Loss : " + h.history.loss[0] + ' / Val_Loss : ' + h.history.val_loss[0]);

        await this.model.save('file://./mastering-ai');
      }
      xs.dispose();
    }
  }

  async predict(wav) {
    // const model = await tf.loadLayersModel(`file://${__dirname}/mastering-ai/model.json`);
    if (this.model === null) this.build();
    const predictBatch = await this.data.predictBatch(wav);
    const prediction = this.model.predict(predictBatch);
    console.log(prediction);
    const mean = prediction.mean(0).dataSync();
    const freq = this.data.invNorm(mean[0], 20, 20000);
    const gain = this.data.invNorm(mean[1], -15.0, 15.0);
    const q = this.data.invNorm(mean[2], 0.1, 24.0);

    return { freq, gain, q };
  }
}

module.exports = new Model();
