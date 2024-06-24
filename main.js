const fs = require('fs');
const path = require('path');
const convnetjs = require('convnetjs');
const sharp = require('sharp');
const seedrandom = require('seedrandom');

// Constants
const IMAGE_SIZE = 128;
const BATCH_SIZE = 32;
const TRAIN_TEST_SPLIT = 0.8;
const EPOCHS = 1;
const LEARNING_RATE = 0.01;
const SEED = 5;

// Set global seeds
seedrandom(SEED);

// Helper function to load images and convert them to Vol objects
async function loadImagesFromFolder(folderPath, label) {
  const files = fs.readdirSync(folderPath);
  const imageElements = []; // Array to store image elements
  for (const file of files) {
    const imgPath = path.join(folderPath, file);
    try {
      const imageBuffer = await sharp(imgPath)
        .resize(IMAGE_SIZE, IMAGE_SIZE)
        .raw()
        .toBuffer();
      const vol = new convnetjs.Vol(IMAGE_SIZE, IMAGE_SIZE, 3);
      for (let i = 0; i < imageBuffer.length; i += 3) {
        vol.w[(i / 3) * 3] = imageBuffer[i] / 255.0; // R
        vol.w[(i / 3) * 3 + 1] = imageBuffer[i + 1] / 255.0; // G
        vol.w[(i / 3) * 3 + 2] = imageBuffer[i + 2] / 255.0; // B
      }
      imageElements.push({ vol, label });
    } catch (error) {
      console.error(`Error processing image ${file}: ${error.message}`);
    }
  }
  return imageElements;
}

// Load all images
(async function () {
  const cats = await loadImagesFromFolder('./cats_small', 1);
  const notCats = await loadImagesFromFolder('./not_cats_small', 0);

  const allImages = cats.concat(notCats);

  // Shuffle and split data into training and testing sets
  allImages.sort(() => Math.random() - 0.5);
  const splitIndex = Math.floor(TRAIN_TEST_SPLIT * allImages.length);
  const trainImages = allImages.slice(0, splitIndex);
  const testImages = allImages.slice(splitIndex);

  // Build the model
  const layerDefs = [];
  layerDefs.push({
    type: 'input',
    out_sx: IMAGE_SIZE,
    out_sy: IMAGE_SIZE,
    out_depth: 3,
  });
  layerDefs.push({
    type: 'conv',
    sx: 3,
    filters: 32,
    stride: 1,
    pad: 2,
    activation: 'relu',
  });
  layerDefs.push({ type: 'pool', sx: 2, stride: 2 });
  layerDefs.push({
    type: 'conv',
    sx: 3,
    filters: 64,
    stride: 1,
    pad: 2,
    activation: 'relu',
  });
  layerDefs.push({ type: 'pool', sx: 2, stride: 2 });
  layerDefs.push({ type: 'fc', num_neurons: 128, activation: 'relu' });
  layerDefs.push({ type: 'softmax', num_classes: 2 });

  const net = new convnetjs.Net();
  net.makeLayers(layerDefs);

  // Train the model
  const trainer = new convnetjs.Trainer(net, {
    method: 'adadelta',
    batch_size: BATCH_SIZE,
    l2_decay: 0.001,
  });

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    let batchLoss = 0;
    for (let i = 0; i < trainImages.length; i += BATCH_SIZE) {
      for (let j = 0; j < BATCH_SIZE && i + j < trainImages.length; j++) {
        const { vol, label } = trainImages[i + j];
        const stats = trainer.train(vol, label);
        batchLoss += stats.loss;
      }
      console.log(
        `Epoch ${epoch + 1}, Batch ${Math.floor(i / BATCH_SIZE) + 1}, Loss: ${
          batchLoss / BATCH_SIZE
        }`
      );
    }
  }

  // Evaluate the model
  let correct = 0;
  for (const { vol, label } of testImages) {
    const prediction = net.forward(vol);
    const predictedLabel = prediction.w[1] > prediction.w[0] ? 1 : 0;
    if (predictedLabel === label) correct++;
  }
  console.log(
    `Test Accuracy: ${((correct / testImages.length) * 100).toFixed(2)}%`
  );
})();
