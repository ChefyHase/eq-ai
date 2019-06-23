const Data = require('../data/data.js');
const fs = require('fs');
const config = require('../config.js');

(async () => {
  const data = new Data();
  await data.separation();
  await data.applyFilter();
  data.fft();
  data.makeDataset();
  console.log(data.dataSets[0][0])
  console.log(data.dataSets[0][1])

  for (let i = 0; i < config.trainEpoches; i++) {
    const filePath = config.dataSetPath + i + '.json';
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

    const buffer = data.dataSets[i];
    const json = JSON.stringify(buffer);
    const readStream = require('streamifier').createReadStream(Buffer.from(json));
    const writeStream = fs.createWriteStream(filePath);
    readStream.pipe(writeStream);
  }
})();
