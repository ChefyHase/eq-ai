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
    const lambda = 0.01;
    const units = 3000;

    const input = tf.input({ shape: [1001] });
    const dense1 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(input);
    const dense2 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense1);
    const dense3 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense2);
    const node1 = tf.layers.add().apply([dense1, dense3]);
    const dense4 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node1);
    const dense5 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense4);
    const dense6 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense5);
    const node2 = tf.layers.add().apply([dense4, dense6]);
    const dense7 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node2);
    const dense8 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense7);
    const dense9 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense8);
    const node3 = tf.layers.add().apply([dense7, dense9]);
    const dense10 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node3);
    const dense11 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense10);
    const dense12 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense11);
    const node4 = tf.layers.add().apply([dense10, dense12]);
    const dense13 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node4);
    const dense14 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense13);
    const dense15 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense14);
    const node5 = tf.layers.add().apply([dense13, dense15]);
    const dense16 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node5);
    const dense17 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense16);
    const dense18 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense17);
    const node6 = tf.layers.add().apply([dense16, dense18]);
    const dense19 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node6);
    const dense20 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense19);
    const dense21 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense20);
    const node7 = tf.layers.add().apply([dense19, dense21]);
    const dense22 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node7);
    const dense23 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense22);
    const dense24 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense23);
    const node8 = tf.layers.add().apply([dense22, dense24]);
    const dense25 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node8);
    const dense26 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense25);
    const dense27 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense26);
    const node9 = tf.layers.add().apply([dense25, dense27]);
    const dense28 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(node9);
    const dense29 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense28);
    const dense30 = tf.layers.dense({
      units: units,
      activation: 'relu',
      kernelInitializer: 'heNormal',
      kernelRegularizer: tf.regularizers.l2({ l2: lambda })
    }).apply(dense29);
    const node10 = tf.layers.add().apply([dense28, dense30]);
    const noise = tf.layers.gaussianNoise({ stddev: 0.3 }).apply(node10);
    const output = tf.layers.dense({ units: 1, activation: 'linear' }).apply(noise);

    const model = tf.model({ inputs: input, outputs: output });

    // model.summary();
    this.model = model;
    return model;
  }

  lossParam(yTrue, yPred) {
    return tf.tidy(() => {
      // console.log(yTrue.arraySync()[0], yPred.arraySync()[0]);
      let lossParam = tf.metrics.meanSquaredError(yTrue, yPred);
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

  r2(yTrue, yPred) {
    return tf.tidy(() => {
      const num = tf.square(yTrue.sub(yPred)).sum();
      const den = tf.square(yTrue.sub(yTrue.mean())).sum();
      const r2 = tf.scalar(1).sub(num.div(den));
      return r2.dataSync();
    });
  }

  rnm(yTrue, yPred) {
    // Good value is 1.253.
    return tf.tidy(() => {
      const rmse = this.rootMeanSquaredError(yTrue, yPred);
      const mse = tf.metrics.meanSquaredError(yTrue, yPred);
      const rnm = rmse.div(mse).mean();
      return rnm.dataSync();
    });
  }

  async acc(yTrue, yPred, thr) {
    const numTotal = yTrue.flatten().shape[0];
    const filter = tf.fill(yTrue.shape, thr);
    const error = yTrue.sub(yPred).abs();
    const filtered = error.lessEqual(filter);
    let numCorrect = await tf.whereAsync(filtered);
    numCorrect = numCorrect.shape[0];

    return numCorrect / numTotal;
  }

  rootMeanSquaredError(yTrue, yPred) {
    return tf.square(yTrue.sub(yPred)).mean().sqrt();
  }

  async train(args) {
    await this.init();

    const numModels = 1;
    const models = [];
    for (let n = 0; n < numModels; n++) models.push(await this.build());
    const optimizer = tf.train.rmsprop(0.00001);
    const batchSize = config.batchSize;

    for (let j = 0; j < args.iterations; j++) {
      let { xs, raw, ys } = this.data.nextBatch(j % (config.trainEpoches - 1));
      let [xTrain, xVal] = this.data.valSplits(xs, 0.3);
      let [rawTrain, rawVal] = this.data.valSplits(raw, 0.3);
      let [ysTrain, ysVal] = this.data.valSplits(ys, 0.3);
      let splitedTrain = ysTrain.split([1, 1, 1], 1);
      let splitedVal = ysVal.split([1, 1, 1], 1);
      let trainLoss;
      let valLoss;
      let vals = [];

      for (let model of models) {
        trainLoss = await optimizer.minimize(() => {
          let pred = model.predict(this.featureExt(xTrain), {
            batchSize: batchSize
          });
          [, , trainLoss] = this.loss(xTrain, rawTrain.arraySync(), splitedTrain[args.model], pred);
          return trainLoss;
        }, true);
        trainLoss = Number(trainLoss.dataSync());

        let val = model.predict(this.featureExt(xVal), {
          batchSize: Math.ceil(batchSize * 0.3)
        });
        vals.push(val);
      }

      let valMean = this.mean(vals);
      [, , valLoss] = this.loss(xVal, rawVal.arraySync(), splitedVal[args.model], valMean);
      valLoss = Number(valLoss.dataSync());
      let valAcc = await this.acc(splitedVal[args.model], valMean, 0.1);
      console.log(
        `Epoch ${j}: trainLoss: ${trainLoss.toFixed(3)} | valLoss: ${valLoss.toFixed(3)} | valAcc: ${valAcc.toFixed(3)}`
      );

      let r2 = this.r2(xVal, valMean);
      let rnm = this.rnm(xVal, valMean);

      console.log(`R2: ${r2} | RMSE/MSE: ${rnm}`);

      this.data.disposer([xs, ys, valMean]);
      this.data.disposer([xTrain, xVal]);
      this.data.disposer([ysTrain, ysVal]);

      fs.appendFileSync(`./${args.modelName}Log.csv`, `${j}, ${trainLoss}, ${valLoss}, ${valAcc}\n`);
      if (args.save) {
        for (let n = 0; n < numModels; n++) await models[n].save(`file://./eq-ai/models/${args.modelName}/model${n}`);
      }

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
      const total = tensors.reduce((acc, cur) => {
        return acc.add(cur);
      });
      return total.div(tf.scalar(tensors.length));
    });
  }
}

module.exports = new Model();
