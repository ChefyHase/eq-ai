const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');
const peaking = require('node-peaking');
const _ = require('lodash');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model = null;
  }

  build(randomSampling) {
    const filter1 = (randomSampling) ? _.random(19, 27) : 16;
    const filter2 = (randomSampling) ? _.random(11, 16) : 16;
    const lambda = (randomSampling) ? _.random(0.34, 0.44) : 0.1;

    const input = tf.input({ shape: [1024] });
    const reshape = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(input);

    const convLow = tf.layers.conv2d({
      filters: filter1,
      kernelSize: [512, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(reshape);
    const convLow2 = tf.layers.conv2d({
      filters: filter2,
      kernelSize: [512, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(convLow);
    const globalAvePoolLow = tf.layers.globalAveragePooling2d({ name: 'globalAverageLow' }).apply(convLow2);
    const fcLow1 = tf.layers.dense({ units: 1024, activation: 'relu' }).apply(globalAvePoolLow);
    const fcLow2 = tf.layers.dense({ units: 1024, activation: 'sigmoid' }).apply(fcLow1);
    const reshapeLow = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(fcLow2);
    const scaleLow = tf.layers.multiply().apply([convLow2, reshapeLow]);
    const addLow = tf.layers.add().apply([reshape, scaleLow]);
    const poolLow = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(addLow);

    const convMid = tf.layers.conv2d({
      filters: filter1,
      kernelSize: [24, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(reshape);
    const convMid2 = tf.layers.conv2d({
      filters: filter2,
      kernelSize: [24, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(convMid);
    const globalAvePoolMid = tf.layers.globalAveragePooling2d({ name: 'globalAverageMid' }).apply(convMid2);
    const fcMid1 = tf.layers.dense({ units: 1024, activation: 'relu' }).apply(globalAvePoolMid);
    const fcMid2 = tf.layers.dense({ units: 1024, activation: 'sigmoid' }).apply(fcMid1);
    const reshapeMid = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(fcMid2);
    const scaleMid = tf.layers.multiply().apply([convMid2, reshapeMid]);
    const addMid = tf.layers.add().apply([reshape, scaleMid]);
    const poolMid = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(addMid);

    const convHigh = tf.layers.conv2d({
      filters: filter1,
      kernelSize: [3, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(reshape);
    const convHigh2 = tf.layers.conv2d({
      filters: filter2,
      kernelSize: [3, 1],
      activation: 'linear',
      kernelInitializer: 'heNormal',
      padding: 'same',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      strides: [1, 1]
    }).apply(convHigh);
    const globalAvePoolHigh = tf.layers.globalAveragePooling2d({ name: 'globalAverageHigh' }).apply(convHigh2);
    const fcHigh1 = tf.layers.dense({ units: 1024, activation: 'relu' }).apply(globalAvePoolHigh);
    const fcHigh2 = tf.layers.dense({ units: 1024, activation: 'sigmoid' }).apply(fcHigh1);
    const reshapeHigh = tf.layers.reshape({ targetShape: [1024, 1, 1] }).apply(fcHigh2);
    const scaleHigh = tf.layers.multiply().apply([convHigh2, reshapeHigh]);
    const addHigh = tf.layers.add().apply([reshape, scaleHigh]);
    const poolHigh = tf.layers.maxPooling2d({ poolSize: [2, 1] }).apply(addHigh);

    const concat = tf.layers.concatenate().apply([poolLow, poolMid, poolHigh]);
    const flatten = tf.layers.globalMaxPooling2d({ name: 'flatten' }).apply(concat);

    const denseOutput = tf.layers.dense({
      units: 3,
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda }),
      activation: 'sigmoid'
    }).apply(flatten);

    const model = tf.model({ inputs: input, outputs: denseOutput });
    model.summary();
    this.model = model;

    return { model: model, hyperparams: [filter1, filter2, lambda] };
  }

  lossParam(yTrue, yPred) {
    return tf.tidy(() => {
      // console.log(yTrue.arraySync()[0], yPred.arraySync()[0]);
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

      // console.log('LossParam: ' + lossParam.mean().dataSync() + ' / LossSound: ' + lossSound.mean().dataSync());

      return [lossParam, lossSound, totalLoss];
    });
  }

  rootMeanSquaredError(yTrue, yPred) {
    return tf.square(yTrue.sub(yPred)).mean().sqrt();
  }

  async train() {
    // this.model = await tf.loadLayersModel('file://./eq-ai-1/model.json');
    const optimizer = tf.train.adam(0.001);

    const batchSize = 64;

    const { hyperparams } = this.build();
    let trainLoss;
    let valLoss;

    for (let j = 0; j < 300; j++) {
      let { xs, raw, ys } = this.data.nextBatch(j % (config.trainEpoches - 1));
      let [xTrain, xVal] = this.data.valSplits(xs, 0.3);
      let [rawTrain, rawVal] = this.data.valSplits(raw, 0.3);
      let [ysTrain, ysVal] = this.data.valSplits(ys, 0.3);

      trainLoss = await optimizer.minimize(() => {
        let pred = this.model.predict(xTrain, {
          batchSize: batchSize
        });
        [, , trainLoss] = this.loss(xTrain, rawTrain.arraySync(), ysTrain, pred);
        return trainLoss;
      }, true);
      trainLoss = Number(trainLoss.dataSync());

      let val = this.model.predict(xVal, {
        batchSize: batchSize
      });
      [, , valLoss] = this.loss(xVal, rawVal.arraySync(), ysVal, val);
      valLoss = Number(valLoss.dataSync());
      await this.model.save('file://./eq-ai');
      console.log(`Score: ` + trainLoss.toFixed(3) + ' / ' + (valLoss).toFixed(3));

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
