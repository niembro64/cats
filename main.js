const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Constants
const IMAGE_SIZE = 128;
const BATCH_SIZE = 32;
const TRAIN_TEST_SPLIT = 0.8;
const EPOCHS = 10;
const USE_SMALL = true;

// Helper function to load images
function loadImagesFromFolder(folderPath, label) {
  const files = fs.readdirSync(folderPath);
  return files.map((file) => ({
    imagePath: path.join(folderPath, file),
    filename: file,
    label,
  }));
}

// Load all images
const cats = loadImagesFromFolder(USE_SMALL ? './cats_small' : './cats', 1);
const notCats = loadImagesFromFolder(
  USE_SMALL ? './not_cats_small' : './not_cats',
  0
);

const allImages = cats.concat(notCats);

// Shuffle and split data into training and testing sets
tf.util.shuffle(allImages);
const splitIndex = Math.floor(TRAIN_TEST_SPLIT * allImages.length);
const trainImages = allImages.slice(0, splitIndex);
const testImages = allImages.slice(splitIndex);

// Helper function to preprocess images
function preprocessImage(imagePath) {
  const buffer = fs.readFileSync(imagePath);
  const image = tf.node.decodeImage(buffer, 3);
  const resizedImage = tf.image.resizeBilinear(image, [IMAGE_SIZE, IMAGE_SIZE]);
  const normalizedImage = resizedImage.div(255.0);
  return normalizedImage;
}

// Create a dataset from image paths
function createDataset(imagePaths, batchSize) {
  const dataset = tf.data.generator(function* () {
    for (let imageInfo of imagePaths) {
      const { imagePath, label, filename } = imageInfo;
      try {
        let imageTensor = preprocessImage(imagePath);
        if (imageTensor.shape.length === 4 && imageTensor.shape[0] === 1) {
          imageTensor = imageTensor.squeeze([0]);
        }
        if (imageTensor.shape.length !== 3 || imageTensor.shape[2] !== 3) {
          console.error(
            `Skipping image: ${filename} due to unexpected shape: ${imageTensor.shape}`
          );
          continue;
        }
        const labelTensor = tf.tensor1d([label]);
        // Check and print shape
        console.log(
          `Processed image: ${filename}, shape: ${imageTensor.shape}`
        );
        yield { xs: imageTensor, ys: labelTensor };
      } catch (err) {
        console.error(
          `Error processing image: ${filename}, error: ${err.message}`
        );
      }
    }
  });
  return dataset.batch(batchSize);
}

// Create training and testing datasets
const trainDataset = createDataset(trainImages, BATCH_SIZE);
const testDataset = createDataset(testImages, BATCH_SIZE);

// Build the model
const model = tf.sequential();

model.add(
  tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the model
model.compile({
  optimizer: tf.train.adam(),
  loss: tf.losses.sigmoidCrossEntropy,
  metrics: ['accuracy'],
});

// Train the model with detailed logging
async function trainModel() {
  const startTime = Date.now();

  await model.fitDataset(trainDataset, {
    epochs: EPOCHS,
    validationData: testDataset,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1} / ${EPOCHS}: loss = ${logs.loss.toFixed(
            4
          )}, accuracy = ${(logs.acc * 100).toFixed(
            2
          )}%, val_loss = ${logs.val_loss.toFixed(4)}, val_accuracy = ${(
            logs.val_acc * 100
          ).toFixed(2)}%`
        );
      },
      onBatchEnd: (batch, logs) => {
        console.log(
          `Batch ${batch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${(
            logs.acc * 100
          ).toFixed(2)}%`
        );
      },
    },
  });

  const endTime = Date.now();
  console.log(
    `Model training complete. Training time: ${
      (endTime - startTime) / 1000
    } seconds`
  );
  await evaluateModel();
  await saveModel();
  await predictOnTestData();
}

// Evaluate the model
async function evaluateModel() {
  const result = await model.evaluateDataset(testDataset);
  const testLoss = result[0].dataSync()[0];
  const testAccuracy = result[1].dataSync()[0];
  console.log(`Test Loss: ${testLoss.toFixed(4)}`);
  console.log(`Test Accuracy: ${(testAccuracy * 100).toFixed(2)}%`);
}

// Save the model
async function saveModel() {
  const savePath = 'file://./my-cat-classifier';
  await model.save(savePath);
  console.log(`Model saved to ${savePath}`);
}

const blockChar = '█';
const dashChar = '─';
const getBarsFromPercent = (percent) => {
  const width = 30;
  const progress = Math.round(width * percent);
  const bar = blockChar.repeat(progress) + dashChar.repeat(width - progress);
  return bar;
};

const red = '\u001b[31m';
const green = '\u001b[32m';
const reset = '\u001b[0m';

// Predict on test data
async function predictOnTestData() {
  for (let imageInfo of testImages) {
    const { imagePath, label, filename } = imageInfo;
    const imageTensor = preprocessImage(imagePath).expandDims();
    const prediction = model.predict(imageTensor);
    const inference = prediction.dataSync()[0];

    const progressBar = getBarsFromPercent(inference);

    console.log(label === 1 ? red : green, `${progressBar} ${reset}`);
  }
}

// Execute training
trainModel().catch((err) => console.error('Error during model training:', err));
