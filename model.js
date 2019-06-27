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
    const droprate = 0.0;
    const input = tf.input({ shape: [1024] });
    const reshape = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({
      filters: 256,
      kernelSize: [128, 1],
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      strides: [4, 1]
    }).apply(reshape);
    const norm1 = tf.layers.batchNormalization().apply(conv1);
    const pool1 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm1);
    const dropOut1 = tf.layers.dropout({ rate: droprate }).apply(pool1);

    const conv2 = tf.layers.conv2d({
      filters: 64,
      kernelSize: [64, 1],
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      padding: 'same'
    }).apply(dropOut1);
    const norm2 = tf.layers.batchNormalization().apply(conv2);
    const pool2 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm2);
    const dropOut2 = tf.layers.dropout({ rate: droprate }).apply(pool2);

    const conv3 = tf.layers.conv2d({
      filters: 64,
      kernelSize: [64, 1],
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      padding: 'same'
    }).apply(dropOut2);
    const norm3 = tf.layers.batchNormalization().apply(conv3);
    const pool3 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm3);
    const dropOut3 = tf.layers.dropout({ rate: droprate }).apply(pool3);

    const conv5 = tf.layers.conv2d({
      filters: 128,
      kernelSize: [64, 1],
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      padding: 'same'
    }).apply(dropOut3);
    const norm5 = tf.layers.batchNormalization().apply(conv5);
    const pool5 = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(norm5);
    const dropOut5 = tf.layers.dropout({ rate: droprate }).apply(pool5);

    const conv6 = tf.layers.conv2d({
      filters: 512,
      kernelSize: [64, 1],
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2',
      padding: 'same'
    }).apply(dropOut5);
    const norm6 = tf.layers.batchNormalization().apply(conv6);
    const pool6 = tf.layers.globalMaxPooling2d({ name: 'pool' }).apply(norm6);
    const dropOut6 = tf.layers.dropout({ rate: droprate }).apply(pool6);

    const dense1 = tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dropOut6);
    const dense1Drop = tf.layers.dropout({ rate: droprate }).apply(dense1);
    const denseOutput = tf.layers.dense({
      units: 3,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: 'l1l2'
    }).apply(dense1Drop);

    const model = tf.model({ inputs: input, outputs: denseOutput });
    model.summary();
    this.model = model;

    return model;
  }

  lossParam(yTrue, yPred) {
    return tf.tidy(() => {
      console.log(yTrue.arraySync()[0], yPred.arraySync()[0]);
      let lossParam = tf.metrics.meanSquaredError(yTrue, yPred);
      return lossParam;
    });
  }

  lossSound(x, raw, yPred) {
    return tf.tidy(() => {
      const sound = x.arraySync();
      const params = yPred.arraySync();
      const buffers = [];
      for (let i = 0; i < sound.length; i++) {
        const buffer = sound[i];
        const param = {
          freq: this.data.lin2log(params[i][0]),
          gain: -1 * this.data.invNorm(params[i][1], -24.0, 24.0),
          q: this.data.invNorm(params[i][2], 0.1, 24.0),
          samplerate: this.data.samplerate
        }
        buffers.push(peaking.peaking(buffer, param));
      }
      const predFiltered = tf.tensor(buffers);
      predFiltered.print()
      raw = tf.tensor(raw);
      const lossSound = tf.metrics.meanSquaredError(raw, predFiltered);
      // raw.sub(predFiltered).abs().mean().print()

      return lossSound;
    });
  }

  loss(x, raw, yTrue, yPred) {
    return tf.tidy(() => {
      const lossParam = this.lossParam(yTrue, yPred);
      const lossSound = this.lossSound(x, raw, yPred);
      const totalLoss = tf.mean(lossParam.add(lossSound));

      console.log('LossParam: ' + lossParam.mean().dataSync() + ' / LossSound: ' + lossSound.mean().dataSync());

      return [lossParam, lossSound, totalLoss];
    });
  }

  async train() {
    this.build();
    // this.model = await tf.loadLayersModel('file://./eq-ai-1/model.json');
    const optimizer = tf.train.adam(0.0002);

    let { xs, raw, ys } = this.data.nextBatch();
    let [xTrain, xVal] = this.data.valSplits(xs, 0.3);
    let [rawTrain, rawVal] = this.data.valSplits(raw, 0.3);
    let [ysTrain, ysVal] = this.data.valSplits(ys, 0.3);

    const batchSize = 500;

    for (let j = 0; j < 5000; j++) {
      let trainLoss = await optimizer.minimize(() => {
        let pred = this.model.predict(xTrain, {
          batchSize: batchSize
        });
        let [lossParam, lossSound, totalLoss] = this.loss(xTrain, rawTrain.arraySync(), ysTrain, pred);
        return totalLoss;
      }, true);
      trainLoss = Number(trainLoss.dataSync());

      let val = this.model.predict(xVal, {
        batchSize: batchSize
      });
      let [, , valLoss] = this.loss(xVal, rawVal.arraySync(), ysVal, val);
      valLoss = Number(valLoss.dataSync());

      console.log(`Epoch ${j + 1}: ` + trainLoss.toFixed(3) + ' / ' + (valLoss).toFixed(3));
      // prevLoss = trainLoss;
      // if (j % 50 === 0 && j !== 0) console.log(await this.predict('./Mixdown.wav'));
      await this.model.save('file://./eq-ai');
      await tf.nextFrame();
    }
  }

  async predict(wav) {
    if (this.model === null) this.model = await tf.loadLayersModel(`file://${__dirname}/eq-ai/model.json`);
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
