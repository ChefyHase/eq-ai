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
    const lambda = 0.01;

    const input = tf.input({ shape: [1024] });
    const reshape = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(input);

    const convLow = tf.layers.conv2d({
      filters: 32,
      kernelSize: [512, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(reshape);
    const poolLow = tf.layers.globalMaxPooling2d({ name: 'poolLow' }).apply(convLow);
    const convMid = tf.layers.conv2d({
      filters: 32,
      kernelSize: [24, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(reshape);
    const poolMid = tf.layers.globalMaxPooling2d({ name: 'poolMid' }).apply(convMid);
    const convHigh = tf.layers.conv2d({
      filters: 32,
      kernelSize: [3, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(reshape);
    const poolHigh = tf.layers.globalMaxPooling2d({ name: 'poolHigh' }).apply(convHigh);

    const concat = tf.layers.concatenate().apply([poolLow, poolMid, poolHigh]);

    const denseOutput = tf.layers.dense({
      units: 3,
      activation: 'linear',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(concat);

    const model = tf.model({ inputs: input, outputs: denseOutput });
    model.summary();
    this.model = model;

    return model;
  }

  lossParam(yTrue, yPred) {
    return tf.tidy(() => {
      console.log(yTrue.arraySync()[0], yPred.arraySync()[0]);
      let lossParam = this.rootMeanSquaredError(yTrue, yPred);
      return lossParam;
    });
  }

  lossSound(x, raw, yTrue, yPred, lossParam) {
    if (lossParam.mean().dataSync() < 0.05) {
      return tf.tidy(() => {
        const params = yPred.arraySync();
        const paramsTrue = yTrue.arraySync();
        const buffers = [];
        for (let i = 0; i < raw.length; i++) {
          const buffer = raw[i];
          const param = {
            freq: this.data.lin2log(params[i][0]),
            gain: this.data.invNorm(paramsTrue[i][1], -24.0, 24.0),
            q: this.data.invNorm(paramsTrue[i][2], 0.1, 24.0),
            samplerate: this.data.samplerate
          }
          buffers.push(peaking.peaking(buffer, param));
        }
        const predFiltered = tf.tensor(buffers);
        // Error to Signal Ratio
        const lossSound = tf.div(tf.abs(raw.sub(predFiltered)).square().sum(), tf.abs(raw).square().sum());
        // const lossSound = tf.metrics.meanSquaredError(x, predFiltered)//.div(tf.scalar(2));
        return lossSound;
      });
    }
    else return tf.zeros(lossParam.shape);
  }

  loss(x, raw, yTrue, yPred) {
    return tf.tidy(() => {
      const lossParam = this.lossParam(yTrue, yPred);
      const lossSound = this.lossSound(x, raw, yTrue, yPred, lossParam);
      const totalLoss = tf.mean(lossParam.add(lossSound));

      console.log('LossParam: ' + lossParam.mean().dataSync() + ' / LossSound: ' + lossSound.mean().dataSync());

      return [lossParam, lossSound, totalLoss];
    });
  }

  rootMeanSquaredError(yTrue, yPred) {
    return tf.square(yTrue.sub(yPred)).mean().sqrt();
  }

  async train() {
    this.build();
    // this.model = await tf.loadLayersModel('file://./eq-ai-1/model.json');
    const optimizer = tf.train.adam(0.0001);

    let { xs, raw, ys } = this.data.nextBatch();
    let [xTrain, xVal] = this.data.valSplits(xs, 0.3);
    let [rawTrain, rawVal] = this.data.valSplits(raw, 0.3);
    let [ysTrain, ysVal] = this.data.valSplits(ys, 0.3);

    const batchSize = 10;

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
