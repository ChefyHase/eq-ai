const model = require('../model.js');
const tf = require('@tensorflow/tfjs-node-gpu');
const decoder = require('wav-decoder');

async function eq(wav, loopnum = 1) {
  let buffer = (typeof wav === 'string') ? await decoder.decode(require('fs').readFileSync(wav)) : wav;

  const params = [];
  for (let n = 0; n < loopnum; ++n) {
    const param = await model.predict(buffer);
    params.push(param);
    buffer.channelData[0] = await model.data.applyFilter(
      buffer,
      {
        freq: param.freq,
        q: param.q,
        gain: -1 * param.gain,
        samplerate: model.data.samplerate
      }
    );
    console.log(n);
  }
  return params;
}

module.exports = eq;
