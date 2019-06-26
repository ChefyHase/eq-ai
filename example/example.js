const eq = require('../index.js');

(async() => {
  let res = await eq('./test1.wav', 1);
  console.log(res);
})()
