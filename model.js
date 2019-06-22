const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model;
  }

  build() {
    const droprate = 0.1;
    const input = tf.input({ shape: [2048] });

    const dense1 = tf.layers.dense({ units: 1000 }).apply(input);
    const activ1 = tf.layers.leakyReLU().apply(dense1);
    const norm1 = tf.layers.batchNormalization().apply(activ1);
    const dropout1 = tf.layers.dropout({ rate: droprate }).apply(norm1);

    const dense2 = tf.layers.dense({ units: 500 }).apply(dropout1);
    const activ2 = tf.layers.leakyReLU().apply(dense2);
    const norm2 = tf.layers.batchNormalization().apply(activ2);
    const dropout2 = tf.layers.dropout({ rate: droprate }).apply(norm2);

    const dense3 = tf.layers.dense({ units: 250 }).apply(dropout2);
    const activ3 = tf.layers.leakyReLU().apply(dense3);
    const norm3 = tf.layers.batchNormalization().apply(activ3);
    const dropout3 = tf.layers.dropout({ rate: droprate }).apply(norm3);

    const dense4 = tf.layers.dense({ units: 100 }).apply(dropout3);
    const activ4 = tf.layers.leakyReLU().apply(dense4);
    const norm4 = tf.layers.batchNormalization().apply(activ4);
    const dropout4 = tf.layers.dropout({ rate: droprate }).apply(norm4);

    const dense5 = tf.layers.dense({ units: 50 }).apply(dropout4);
    const activ5 = tf.layers.leakyReLU().apply(dense5);
    const norm5 = tf.layers.batchNormalization().apply(activ5);
    const dropout5 = tf.layers.dropout({ rate: droprate }).apply(norm5);

    const dense6 = tf.layers.dense({ units: 25 }).apply(dropout5);
    const activ6 = tf.layers.leakyReLU().apply(dense6);
    const norm6 = tf.layers.batchNormalization().apply(activ6);
    const dropout6 = tf.layers.dropout({ rate: droprate }).apply(norm6);

    const dense7 = tf.layers.dense({ units: 10 }).apply(dropout6);
    const activ7 = tf.layers.leakyReLU().apply(dense7);
    const norm7 = tf.layers.batchNormalization().apply(activ7);
    const dropout7 = tf.layers.dropout({ rate: droprate }).apply(norm7);

    const denseOutput = tf.layers.dense({ units: 4 }).apply(dropout7);
    const activOutput = tf.layers.leakyReLU().apply(denseOutput);

    const model = tf.model({ inputs: input, outputs: activOutput });
    model.summary();
    this.model = model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0001);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError', metrics: ['accuracy'] });

    for (let i = 0; i < config.trainEpoches; ++i) {
      const { xs, ys } = this.data.nextBatch();

      const h = await this.model.fit(xs, ys, {
        batchSize: 500,
        epochs: 500,
        shuffle: true,
        validationSplit: 0.4
      });

      console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
      await this.model.save('file://mastering-ai');

      xs.dispose();
      ys.dispose();

      // let buffer = await data.separateSound(__dirname + '/test.wav');
      // let xs = data.fft(tf.tensor(buffer));
    }
  }
}

module.exports = new Model();
