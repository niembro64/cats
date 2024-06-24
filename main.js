const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const seedrandom = require('seedrandom');
const tf = require('@tensorflow/tfjs-node'); // Add TensorFlow.js Node.js backend

// Constants
const IMAGE_SIZE = 256; // Update image size to match ResNet input size
const BATCH_SIZE = 32;
const TRAIN_TEST_SPLIT = 0.8;
const EPOCHS = 10;
const LEARNING_RATE = 0.0001; // Further reduced learning rate
const SEED = 5;
const USE_SMALL = true;

// Set global seeds
const rng = seedrandom(SEED);
Math.random = rng;

// Helper function to load images and convert them to tf.Tensor objects
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
      const imageTensor = tf
        .tensor3d(new Uint8Array(imageBuffer), [IMAGE_SIZE, IMAGE_SIZE, 3])
        .div(255.0);
      imageElements.push({ tensor: imageTensor, label, fileName: file });
    } catch (error) {
      console.error(`Error processing image ${file}: ${error.message}`);
    }
  }
  return imageElements;
}

// Shuffle function that uses the seed
function shuffleArray(array, rng) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// Build the ResNet model
const modelResNet = () => {
  const input = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3] });
  const conv1_filter = tf.layers
    .conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 2,
      activation: 'relu',
      padding: 'same',
      kernelInitializer: 'glorotNormal',
    })
    .apply(input);
  const conv1 = tf.layers
    .maxPooling2d({
      poolSize: [3, 3],
      strides: [2, 2],
      padding: 'same',
    })
    .apply(conv1_filter);

  // conv 2
  const residual2 = residualBlock(conv1, 16, true);

  // conv3
  const residual3 = residualBlock(residual2, 32);

  // conv4
  const residual4 = residualBlock(residual3, 64);

  // conv5
  const residual5 = residualBlock(residual4, 128);
  const conv5 = tf.layers
    .avgPool2d({
      poolSize: [8, 8],
      strides: [1, 1],
    })
    .apply(residual5);

  const flatten = tf.layers.flatten().apply(conv5);
  const dropout = tf.layers.dropout({ rate: 0.5 }).apply(flatten);
  const dense = tf.layers
    .dense({
      units: 2,
      kernelInitializer: 'glorotNormal',
      activation: 'softmax', // softmax for categorical / relu
    })
    .apply(dropout);

  return tf.model({
    inputs: input,
    outputs: dense,
  });
};

// Batch normalisation and ReLU always go together, let's add them to the separate function
const batchNormRelu = (input) => {
  const batch = tf.layers.batchNormalization().apply(input);
  return tf.layers.reLU().apply(batch);
};

// Residual block
const residualBlock = (input, filters, noDownSample = false) => {
  let stride = noDownSample ? 1 : 2;

  const filter1 = tf.layers
    .separableConv2d({
      kernelSize: 3,
      filters,
      activation: 'relu',
      padding: 'same',
      strides: stride,
      depthwiseInitializer: 'glorotNormal',
      pointwiseInitializer: 'glorotNormal',
    })
    .apply(input);
  const filter1norm = batchNormRelu(filter1);

  const filter2 = tf.layers
    .separableConv2d({
      kernelSize: 3,
      filters,
      activation: 'relu',
      padding: 'same',
      depthwiseInitializer: 'glorotNormal',
      pointwiseInitializer: 'glorotNormal',
    })
    .apply(filter1norm);
  const dropout = tf.layers.dropout({ rate: 0.3 }).apply(filter2);
  const batchNorm = batchNormRelu(dropout);

  let inputAdjusted = input;
  if (!noDownSample) {
    inputAdjusted = tf.layers
      .conv2d({
        kernelSize: 1,
        filters,
        strides: stride,
        padding: 'same',
        kernelInitializer: 'glorotNormal',
      })
      .apply(input);
  }

  // Residual connection - here we sum up the adjusted input and the result of 2 convolutions
  const residual = tf.layers.add().apply([inputAdjusted, batchNorm]);
  return residual;
};

// Load all images
(async function () {
  const pathCats = './cats';
  const pathCatsSmall = './cats_small';
  const pathNotCats = './not_cats';
  const pathNotCatsSmall = './not_cats_small';

  const cats = await loadImagesFromFolder(
    USE_SMALL ? pathCatsSmall : pathCats,
    1
  );
  const notCats = await loadImagesFromFolder(
    USE_SMALL ? pathNotCatsSmall : pathNotCats,
    0
  );

  const allImages = cats.concat(notCats);

  // Shuffle and split data into training and testing sets
  shuffleArray(allImages, rng);
  const splitIndex = Math.floor(TRAIN_TEST_SPLIT * allImages.length);
  const trainImages = allImages.slice(0, splitIndex);
  const testImages = allImages.slice(splitIndex);

  // Prepare training and testing datasets
  const trainXs = tf.stack(trainImages.map(({ tensor }) => tensor));
  const trainYs = tf.oneHot(
    trainImages.map(({ label }) => label),
    2
  );
  const testXs = tf.stack(testImages.map(({ tensor }) => tensor));
  const testYs = tf.oneHot(
    testImages.map(({ label }) => label),
    2
  );

  // Build and compile the model
  const model = modelResNet();
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Train the model
  await model.fit(trainXs, trainYs, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
  });

  // Save the trained model
  await model.save('file://./trained_model');

  // Evaluate the model on test data
  const evalResult = model.evaluate(testXs, testYs);
  console.log(`Test loss: ${evalResult[0].dataSync()}`);
  console.log(`Test accuracy: ${evalResult[1].dataSync()}`);
})();
