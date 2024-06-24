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
const LEARNING_RATE = 0.0001;
const SEED = 5;
const MODEL_SAVE_PATH = './trained_model.json';

// Set global seeds
const rng = seedrandom(SEED);
Math.random = rng;
convnetjs.randf = (a, b) => a + (b - a) * rng();
convnetjs.randi = (a, b) => Math.floor(a + (b - a) * rng());
convnetjs.randn = (mu, std) =>
  mu + std * (Math.sqrt(-2 * Math.log(rng())) * Math.cos(2 * Math.PI * rng()));

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
      imageElements.push({ vol, label, fileName: file });
    } catch (error) {
      console.error(`Error processing image ${file}: ${error.message}`);
    }
  }
  return imageElements;
}

// Function to save the network to a file
function saveNetwork(net, filePath) {
  const json = net.toJSON();
  fs.writeFileSync(filePath, JSON.stringify(json));
  console.log(`Model saved to ${filePath}`);
}

// Shuffle function that uses the seed
function shuffleArray(array, rng) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

const red = '\x1b[31m';
const green = '\x1b[32m';
const reset = '\x1b[0m';

const blockChar = '█';
const dashChar = '─';
const getBarsFromPercent = (percent) => {
  percent = Math.min(1, Math.max(0, percent));

  const width = 30;
  const progress = Math.round(width * percent);
  const bar =
    blockChar.repeat(progress) + dashChar.repeat(Math.max(0, width - progress));
  return bar;
};

// Load all images
(async function () {
  const cats = await loadImagesFromFolder('./cats_small', 1);
  const notCats = await loadImagesFromFolder('./not_cats_small', 0);

  const allImages = cats.concat(notCats);

  // Shuffle and split data into training and testing sets
  shuffleArray(allImages, rng);
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
    for (let i = 0; i < trainImages.length; i += BATCH_SIZE) {
      for (let j = 0; j < BATCH_SIZE && i + j < trainImages.length; j++) {
        const { vol, label, fileName } = trainImages[i + j];
        const stats = trainer.train(vol, label);
        const loss = stats.loss;

        print(label, loss, fileName);
      }
    }
  }

  // Save the trained model
  saveNetwork(net, MODEL_SAVE_PATH);

  for (const { vol, label, fileName } of testImages) {
    const prediction = net.forward(vol);

    const p = prediction.w[0];
    print(label, p, fileName);
  }
  for (const { vol, label } of trainImages) {
    const prediction = net.forward(vol);
    const p = prediction.w[0];
    print(label, p);
  }
})();

const print = (label, prediction, fileName) => {
  const bars = getBarsFromPercent(prediction);
  const color = label === 1 ? red : green;

  console.log(
    color + prediction.toFixed(4) + ' ' + bars + ' ' + fileName + reset
  );
};
