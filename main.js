const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Constants
const IMAGE_SIZE = 128;
const BATCH_SIZE = 32;
const TRAIN_TEST_SPLIT = 0.8;
const EPOCHS = 10;

// Helper function to load images
function loadImagesFromFolder(folderPath, label) {
  const files = fs.readdirSync(folderPath);
  return files.map((file) => ({
    imagePath: path.join(folderPath, file),
    filename: file,
    label,
  }));
}

const useSmall = false;

// Load all images
const cats = loadImagesFromFolder(useSmall ? './cats_small' : './cats', 1);
const notCats = loadImagesFromFolder(
  useSmall ? './not_cats_small' : './not_cats',
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
      const { imagePath, label } = imageInfo;
      const imageTensor = preprocessImage(imagePath);
      const labelTensor = tf.tensor1d([label]);
      yield { xs: imageTensor, ys: labelTensor };
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

// Train the model
async function trainModel() {
  await model.fitDataset(trainDataset, {
    epochs: EPOCHS,
    validationData: testDataset,
  });
  console.log('Model training complete.');
  evaluateModel();
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

// Predict on test data
async function predictOnTestData() {
  const predictions = [];
  const labels = [];
  const filenames = [];

  for (let imageInfo of testImages) {
    const { imagePath, label, filename } = imageInfo;
    const imageTensor = preprocessImage(imagePath).expandDims();
    const prediction = model.predict(imageTensor);
    const predictedLabel = prediction.dataSync()[0] > 0.5 ? 1 : 0;

    predictions.push(predictedLabel);
    labels.push(label);
    filenames.push(filename);
  }

  // Output detailed prediction results
  for (let i = 0; i < predictions.length; i++) {
    console.log(
      `Filename: ${filenames[i]} | Predicted: ${predictions[i]} | Actual: ${labels[i]}`
    );
  }

  // Calculate accuracy
  let correct = 0;
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === labels[i]) {
      correct++;
    }
  }
  const accuracy = (correct / predictions.length) * 100;
  console.log(`Prediction Accuracy on Test Data: ${accuracy.toFixed(2)}%`);
}

// Execute training
trainModel().catch((err) => console.error('Error during model training:', err));
