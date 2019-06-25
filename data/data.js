const encoder = require('wav-encoder');
const decoder = require('wav-decoder');
const peaking = require('node-peaking');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const _ = require('lodash');
const config = require('../config');


class Data {
  constructor(args) {
    this.samplerate = 44100;
    this.shortTimeSamples = 512; // Math.pow(2, 16);

    this.sounds = [];
    this.filterParams = [];
    this.dataSets = [];

    this.index = 0;
    this.batchIndex = 0;
  }

  randomFilterParam() {
    const params = {
      freq: _.random(20, 20000),
      q: _.random(0.1, 24.0, true),
      gain: _.random(-24.0, 24.0, true),
      samplerate: this.samplerate
    }
    return params;
  }

  async separateSound(sound, chunnel = 0) {
    let soundBuffer;
    if (typeof sound === 'object') {
      soundBuffer = sound;
    }
    else {
      const soundFilaPeth = path.resolve(sound);
      soundBuffer = await decoder.decode(fs.readFileSync(soundFilaPeth));
    }
    soundBuffer = soundBuffer.channelData[chunnel];

    const shortTimeSounds = _.chunk(soundBuffer, this.shortTimeSamples);
    shortTimeSounds.pop();

    return shortTimeSounds;
  }

  async separation() {
    for (let i = 0; i < config.numSamples; i++) {
      let p = path.join(__dirname, 'sounds', i + '.wav');
      let buffers = await this.separateSound(p, 0);
      // random selection
      let index = 0;
      let randIndex = Array(buffers.length).fill(0);
      randIndex = randIndex.map((elem) => {
        return index++;
      });
      randIndex = _.shuffle(randIndex).slice(0, config.samplesPerSong);
      let randL = [];
      for (let i = 0; i < randIndex.length; i++) {
        randL.push(buffers[randIndex[i]]);
      }

      if (config.varbose) console.log(i + ' / ' + config.numSamples);

      this.sounds.push(...randL);
    }
  }

  async applyFilter(data = null, param = null) {
    if (data && param) {
      const filterParam = param;
      const separated = await this.separateSound(data);
      const filtered = [];
      for (let buffer of separated) {
        filtered.push(...peaking.peaking(buffer, filterParam))
      }
      return filtered;
    }
    else {
      for (let i = 0; i < this.sounds.length; i++) {
        const filterParam = this.randomFilterParam();
        const filtered = peaking.peaking(this.sounds[i], filterParam);

        this.sounds[i] = filtered;
        this.filterParams.push(filterParam);
      }
    }
  }

  makeDataset() {
    for (let n = 0; n < config.trainEpoches; n++) {
      const xBatch = [];
      const labelBatch = [];
      for (let i = 0; i < config.batchSize; i++) {
        xBatch.push(...this.sounds[this.batchIndex]);
        labelBatch.push([
          this.norm(this.filterParams[this.batchIndex].freq, 20, 20000),
          this.norm(this.filterParams[this.batchIndex].gain, -24.0, 24.0),
          this.norm(this.filterParams[this.batchIndex].q, 0.1, 24.0)
        ]);
        this.batchIndex++;
      }
      this.dataSets.push([xBatch, labelBatch]);
    }
  }

  nextBatch() {
    this.loadDataset(this.index);
    this.index++;
    const ys = [];
    for (let j = 0; j < 24; j++) {
      let tens = [];
      for (let i = 0; i < this.dataSets[1].length; i++) {
        tens.push([this.dataSets[1][i][j]]);
      }
      ys.push(tf.tensor(tens));
    }
    return {
      xs: this.normTens(tf.tensor(this.dataSets[0], null, 'float32')),
      ys: ys
    }
  }

  async predictBatch(wav) {
    const buffer = await this.separateSound(wav);
    const ts = this.normTens(tf.tensor(buffer));
    return ts;
  }

  loadDataset(index) {
    const filePath = config.dataSetPath + index + '.json';
    let json = JSON.parse(fs.readFileSync(filePath));
    const soundBuffer = _.chunk(json[0], this.shortTimeSamples);
    const params = [];
    for (let param of json[1]) {
      let freqBin = this.dec2Bin(param[0])
      let qBin = this.dec2Bin(param[1])
      let gainBin = this.dec2Bin(param[2])
      params.push([...freqBin, ...qBin, ...gainBin]);
    }
    this.dataSets = [soundBuffer, params];
  }

  dec2Bin(dec) {
    let bin = parseInt(255 * dec, 10).toString(2).split('');
    bin = bin.reverse();
    let output = Array(8).fill(0);
    for (let i = 0; i < bin.length; i++) {
      output[i] = Number(bin[i]);
    }
    output.reverse();
    return output;
  }

  disposer(tensors) {
    tensors.forEach((elem) => {
      elem.dispose();
    });
  }

  norm(x, min, max) {
    return (x - min) / (max - min);
  }

  normTens(xs) {
    return tf.div(xs.sub(xs.min()), xs.max().sub(xs.min()));
  }

  invNorm(y, min, max) {
    return y * (max - min) + min
  }
}

module.exports = Data;
