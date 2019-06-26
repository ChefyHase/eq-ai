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
        filters: 16,
        kernelSize: [3, 1],
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'heNormal',
        kernelRegularizer: 'l1l2',
        strides: [2, 1]
      }).apply(reshape);
      const conv2 = tf.layers.conv2d({
        filters: 16,
        kernelSize: [3, 1],
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'heNormal',
        kernelRegularizer: 'l1l2',
        strides: [2, 1]
      }).apply(conv1);
      const norm1 = tf.layers.batchNormalization().apply(conv2);
      const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm1);

      const flatten = tf.layers.flatten().apply(pool1)

      const dense1 = tf.layers.dense({
        units: 1024,
        activation: 'sigmoid',
        kernelInitializer: 'heNormal',
        kernelRegularizer: 'l1l2'
      }).apply(flatten);
      const denseOutput = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        kernelInitializer: 'heNormal',
        kernelRegularizer: 'l1l2'
      }).apply(dense1);

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

    const optimizer = tf.train.adam(0.005);
    this.model.compile({ optimizer: optimizer, loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    for (let j = 0; j < config.trainEpoches; ++j) {
      let { xs, ys } = this.data.nextBatch();
      for (let i = 0; i < 50; i++) {
        const h = await this.model.fit(xs, ys, {
          batchSize: 64,
          epochs: 1,
          shuffle: true,
          validationSplit: 0.3,
          verbose: 0
        });

        console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
        console.log("Val_Loss after Epoch " + i + " : " + h.history.val_loss[0]);

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
