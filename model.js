const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');
const peaking = require('node-peaking');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model = null;
  }

  build() {
    const droprate = 0;
    const input = tf.input({ shape: [1024] });
    const reshape = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({
      filters: 128,
      kernelSize: [128, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(reshape);
    const conv2 = tf.layers.conv2d({
      filters: 128,
      kernelSize: [128, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(conv1);
    const norm1 = tf.layers.batchNormalization().apply(conv2);
    const pool = tf.layers.globalMaxPooling2d({ name: 'pool' }).apply(norm1)

    const dense1 = tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(pool);
    const dense2 = tf.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dense1);
    const denseOutput = tf.layers.dense({
      units: 3,
      activation: 'sigmoid',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dense2);

    const model = tf.model({ inputs: input, outputs: denseOutput });
    model.summary();
    this.model = model;

    return model;
  }

  lossParam(yTrue, yPred) {
    return tf.tidy(() => {
      let lossParam = tf.metrics.meanSquaredError(yTrue, yPred);
      return lossParam;
    });
  }

  lossSound(x, yPred) {
    return tf.tidy(() => {
      const sound = x.arraySync();
      const params = yPred.arraySync();
      const buffers = [];
      for (let i = 0; i < sound.length; i++) {
        const buffer = sound[i];
        const param = {
          freq: this.data.invNorm(params[i][0], 20, 20000),
          q: this.data.invNorm(params[i][2], 0.1, 24.0),
          gain: -1 * this.data.invNorm(params[i][1], -24.0, 24.0),
          samplerate: this.data.samplerate
        }
        buffers.push(buffer, param);
      }
      const predFiltered = tf.tensor(buffers[0]);
      const lossSound = tf.metrics.meanSquaredError(x, predFiltered);

      return lossSound;
    });
  }

  loss(x, yTrue, yPred) {
    return tf.tidy(() => {
      const lossParam = this.lossParam(yTrue, yPred);
      const lossSound = this.lossSound(x, yPred);
      const totalLoss = tf.mean(lossParam.add(lossSound));
      return totalLoss;
    });
  }

  async train() {
    this.build();
    const optimizer = tf.train.adam(0.001);

    let { xs, ys } = this.data.nextBatch();
    for (let j = 0; j < 500; j++) {
      let trainLoss = await optimizer.minimize(() => {
        let pred = this.model.predict(xs);
        return this.loss(xs, ys, pred);
      }, true);
      trainLoss = Number(trainLoss.dataSync());
      console.log(trainLoss);
      const res = await this.predict('./Mixdown.wav');
      console.log(res);
      await this.model.save('file://./eq-ai');
      await tf.nextFrame();
    }
  }

  async predict(wav) {
    // const model = await tf.loadLayersModel(`file://${__dirname}/mastering-ai/model.json`);
    if (this.model === null) this.build();
    const predictBatch = await this.data.predictBatch(wav);
    const prediction = this.model.predict(predictBatch);
    const mean = prediction.mean(0).dataSync();
    const freq = this.data.invNorm(mean[0], 20, 20000);
    const gain = this.data.invNorm(mean[1], -24.0, 24.0);
    const q = this.data.invNorm(mean[2], 0.1, 24.0);

    return { freq, gain, q };
  }
}

module.exports = new Model();
