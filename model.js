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
    const droprate = 0.2;
    const input = tf.input({ shape: [1024] });
    const reshape = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({
      filters: 16,
      kernelSize: [64, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(reshape);
    const conv2 = tf.layers.conv2d({
      filters: 16,
      kernelSize: [64, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(conv1);
    const norm1 = tf.layers.batchNormalization().apply(conv2);
    const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm1);
    const dropOut1 = tf.layers.dropout({ rate: droprate }).apply(pool1);

    const conv3 = tf.layers.conv2d({
      filters: 256,
      kernelSize: [32, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(dropOut1);
    const conv4 = tf.layers.conv2d({
      filters: 256,
      kernelSize: [32, 1],
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [2, 1]
    }).apply(conv3);
    const norm2 = tf.layers.batchNormalization().apply(conv4);
    const pool2 = tf.layers.globalMaxPooling2d({ name: 'pool' }).apply(norm2);
    const dropOut2 = tf.layers.dropout({ rate: droprate }).apply(pool2);

    const dense1 = tf.layers.dense({
      units: 126,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropOut2);
    const dropOut3 = tf.layers.dropout({ rate: droprate }).apply(dense1);
    const dense2 = tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropOut3);
    const dropOut4 = tf.layers.dropout({ rate: droprate }).apply(dense2);
    const dense3 = tf.layers.dense({
      units: 32,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropOut4);
    const dropOut5 = tf.layers.dropout({ rate: droprate }).apply(dense3);
    const dense4 = tf.layers.dense({
      units: 16,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropOut5);
    const dropOut6 = tf.layers.dropout({ rate: droprate }).apply(dense4);
    const denseOutput = tf.layers.dense({
      units: 3,
      activation: 'sigmoid',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropOut6);

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

  lossSound(x, raw, yPred) {
    return tf.tidy(() => {
      const params = yPred.arraySync();
      const buffers = [];
      for (let i = 0; i < raw.length; i++) {
        const buffer = raw[i];
        const param = {
          freq: this.data.lin2log(params[i][0]),
          q: this.data.invNorm(params[i][2], 0.1, 24.0),
          gain: -1 * this.data.invNorm(params[i][1], -24.0, 24.0),
          samplerate: this.data.samplerate
        }
        buffers.push(peaking.peaking(buffer, param));
      }
      const predFiltered = tf.tensor(buffers);
      const diff = predFiltered.sub(x);
      const lossSound = tf.metrics.meanSquaredError(tf.zeros(diff.shape), diff);

      return lossSound;
    });
  }

  loss(x, raw, yTrue, yPred) {
    return tf.tidy(() => {
      const lossParam = this.lossParam(yTrue, yPred);
      const lossSound = this.lossSound(x, raw, yPred);
      const totalLoss = lossSound.mean().mul(tf.scalar(1 - 0.3)).add(lossParam.mean().mul(tf.scalar(0.3)));

      // console.log('LossParam: ' + lossParam.mean().dataSync() + ' / LossSound: ' + lossSound.mean().dataSync());

      return totalLoss;
    });
  }

  async train() {
    this.build();
    // this.model = await tf.loadLayersModel('file://./eq-ai/model.json');
    const optimizer = tf.train.adam(0.0005);

    let { xs, raw, ys } = this.data.nextBatch();
    let [xTrain , xVal] = this.data.valSplits(xs, 0.3);
    let [rawTrain , rawVal] = this.data.valSplits(raw, 0.3);
    let [ysTrain , ysVal] = this.data.valSplits(ys, 0.3);

    const batchSize = 200;

    let prevLoss = 0;
    for (let j = 0; j < 5000; j++) {
      let trainLoss = await optimizer.minimize(() => {
        let pred = this.model.predict(this.data.normTens(xTrain), {
          batchSize: batchSize
        });
        return this.loss(xTrain, rawTrain.arraySync(), ysTrain, pred);
      }, true);
      trainLoss = Number(trainLoss.dataSync());

      let val = this.model.predict(this.data.normTens(xVal), {
        batchSize: batchSize
      });
      let valLoss = this.loss(xVal, rawVal.arraySync(), ysVal, val);
      valLoss = Number(valLoss.dataSync());

      console.log(`Epoch ${j + 1}: ` + trainLoss + ' / ' + valLoss);
      // prevLoss = trainLoss;
      // if (j % 50 === 0 && j !== 0) console.log(await this.predict('./Mixdown.wav'));
      await this.model.save('file://./eq-ai-1');
      await tf.nextFrame();
    }
  }

  async predict(wav) {
    if (this.model === null) this.model = await tf.loadLayersModel(`file://${__dirname}/eq-ai-1/model.json`);
    const predictBatch = await this.data.predictBatch(wav);
    const prediction = this.model.predict(predictBatch);
    const mean = prediction.mean(0).dataSync();
    const freq = this.data.lin2log(mean[0]);
    const gain = this.data.invNorm(mean[1], -24.0, 24.0);
    const q = this.data.invNorm(mean[2], 0.1, 24.0);

    return { freq, gain, q };
  }
}

module.exports = new Model();
