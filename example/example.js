const eq = require('../index.js');

(async() => {
  let res = await eq('./Mixdown.wav', 5);
  console.log(res);
})()
