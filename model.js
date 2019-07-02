const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');
const peaking = require('node-peaking');
const _ = require('lodash');
const fs = require('fs');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model = null;
    this.imageNet;
  }

  async init() {
    this.imageNet = await tf.loadGraphModel(
      'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2',
      { fromTFHub: true }
    );
  }

  async build(randomSampling) {
    const lambda = 0.03;

    const input = tf.input({ shape: [1001] });
    const dense1 = tf.layers.dense({
      units: 2450,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(input);
    const norm1 = tf.layers.batchNormalization().apply(dense1);
    const dense2 = tf.layers.dense({
      units: 1200,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(norm1);
    const norm2 = tf.layers.batchNormalization().apply(dense2);
    const dense3 = tf.layers.dense({
      units: 500,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(norm2);
    const output = tf.layers.dense({ units: 3, activation: 'linear' }).apply(dense3);

    const model = tf.model({ inputs: input, outputs: output });

    // model.summary();
    this.model = model;
    return model;
  }

  lossParam(yTrue, yPred) {
    return tf.tidy(() => {
      // console.log(yTrue.arraySync()[0], yPred.arraySync()[0]);
      let lossParam = this.rootMeanSquaredError(yTrue, yPred);
      return lossParam;
    });
  }

  lossSound(x, raw, yTrue, yPred, lossParam) {
    if (lossParam.mean().dataSync() < 0.0) {
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
        // const lossSound = tf.div(tf.abs(raw.sub(predFiltered)).square().sum(), tf.abs(raw).square().sum());
        const lossSound = tf.metrics.meanSquaredError(x, predFiltered)//.div(tf.scalar(2));
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

  async acc(yTrue, yPred, thr) {
    const numTotal = yTrue.flatten().shape[0];
    const filter = tf.fill(yTrue.shape, thr);
    const error = yTrue.sub(yPred).abs();
    let numCorrect = await tf.whereAsync(error.lessEqual(filter));
    numCorrect = numCorrect.flatten().shape[0];

    return numCorrect / numTotal;
  }

  rootMeanSquaredError(yTrue, yPred) {
    return tf.square(yTrue.sub(yPred)).mean().sqrt();
  }

  async train() {
    await this.init();

    const numModels = 1;
    const models = [];
    for (let n = 0; n < numModels; n++) models.push(await this.build());
    const optimizer = tf.train.adam(0.0001);
    const batchSize = config.batchSize;

    for (let j = 0; j < 100; j++) {
      let { xs, raw, ys } = this.data.nextBatch(j % (config.trainEpoches - 1));
      let [xTrain, xVal] = this.data.valSplits(xs, 0.3);
      let [rawTrain, rawVal] = this.data.valSplits(raw, 0.3);
      let [ysTrain, ysVal] = this.data.valSplits(ys, 0.3);
      let trainLoss;
      let valLoss;
      let vals = [];

      for (let model of models) {
        trainLoss = await optimizer.minimize(() => {
          let pred = model.predict(this.featureExt(xTrain), {
            batchSize: batchSize
          });
          [, , trainLoss] = this.loss(xTrain, rawTrain.arraySync(), ysTrain, pred);
          return trainLoss;
        }, true);
        trainLoss = Number(trainLoss.dataSync());

        let val = model.predict(this.featureExt(xVal), {
          batchSize: batchSize
        });
        vals.push(val);
      }

      let valMean = this.mean(vals);
      [, , valLoss] = this.loss(xVal, rawVal.arraySync(), ysVal, valMean);
      valLoss = Number(valLoss.dataSync());
      let valAcc = await this.acc(ysVal, valMean, 0.1);
      console.log(
        `Epoch ${j}: trainLoss: ${trainLoss.toFixed(3)} | valLoss: ${valLoss.toFixed(3)} | valAcc: ${valAcc.toFixed(3)}`
      );

      this.data.disposer([xs, ys, valMean]);
      this.data.disposer([xTrain, xVal]);
      this.data.disposer([ysTrain, ysVal]);

      fs.appendFileSync('./output.csv', `${j}, ${trainLoss}, ${valLoss}, ${valAcc}\n`);
      for (let n = 0; n < numModels; n++) await models[n].save('file://./eq-ai/model' + n);

      await tf.nextFrame();
    }
  }

  async predict(wav) {
    if (this.model === null) {
      await this.init();
      this.model = await tf.loadLayersModel(`file://${__dirname}/eq-ai/model.json`)
    };

    const predictBatch = await this.data.predictBatch(wav);
    const prediction = this.model.predict(this.featureExt(predictBatch));
    const mean = prediction.mean(0).dataSync();
    const freq = this.data.lin2log(mean[0]);
    const gain = this.data.invNorm(mean[1], -24.0, 24.0);
    const q = this.data.invNorm(mean[2], 0.1, 24.0);

    return { freq, gain, q };
  }

  featureExt(x) {
    return tf.tidy(() => {
      const batchSize = x.shape[0];
      x = tf.concat([x, x, x]);
      const inputReshaped = tf.image.resizeBilinear(
        tf.reshape(x, [batchSize, 32, 32, 3]),
        [224, 224]
      );
      const pred = this.imageNet.predict(inputReshaped);
      return pred;
    })
  }

  mean(tensors) {
    return tf.tidy(() => {
      let total = tf.zeros(tensors[0].shape);
      tensors.forEach((tensor) => {
        total = total.add(tensor);
      });
      return total.div(tf.scalar(tensors.length));
    });
  }
}

module.exports = new Model();
