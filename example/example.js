const eq = require('../index.js');

(async() => {
  let res = await eq('./test1.wav', 3);
  console.log(res);
})()
