const cv = require('opencv4nodejs');
const { loadModel } = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { promisify } = require('util');

// Load the Keras model
const readFileAsync = promisify(fs.readFile);
const modelPath = 'best_model.h5';
const loadKerasModel = async (modelPath) => {
  const modelBuffer = await readFileAsync(modelPath);
  return await loadModel(tf.io.browserFiles([modelBuffer]));
};
const model = await loadKerasModel(modelPath);

// Load the face detection cascade classifier
const faceCascadeClassifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_DEFAULT);

// Start the webcam capture
const cap = new cv.VideoCapture(0);
const FPS = 30;

while (true) {
  const frame = cap.read();

  // Resize the frame to improve performance
  const resizedFrame = frame.resize(0.5, 0.5);

  // Convert the image to grayscale
  const grayImg = resizedFrame.bgrToGray();

  // Detect faces in the image
  const faces = faceCascadeClassifier.detectMultiScale(grayImg).objects;

  // Process each face in the image
  for (const face of faces) {
    // Draw a rectangle around the face
    const { x, y, width, height } = face;
    const color = new cv.Vec(255, 0, 0);
    const thickness = 7;
    resizedFrame.drawRectangle(new cv.Point2(x, y), new cv.Point2(x + width, y + height), color, thickness);

    // Extract the region of interest (ROI) i.e. the face from the image
    const roi = resizedFrame.getRegion(new cv.Rect(x, y, width, height));

    // Preprocess the ROI image
    const processedRoi = roi.resize(new cv.Size(224, 224)).div(255.0);

    // Convert the ROI image to a tensor
    const tensor = tf.expandDims(processedRoi.getDataAsArray(), 0);

    // Make a prediction using the Keras model
    const predictions = model.predict(tensor).dataSync();

    // Find the index of the maximum value in the predictions array
    const maxIndex = predictions.indexOf(Math.max(...predictions));

    // Map the index to an emotion label
    const emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
    const predictedEmotion = emotions[maxIndex];

    // Add the predicted emotion label to the frame
    const fontScale = 1;
    const fontColor = new cv.Vec(0, 0, 255);
    const thickness = 2;
    resizedFrame.putText(predictedEmotion, new cv.Point2(x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, thickness);
  }

  // Show the frame in a window
  const winName = 'Facial emotion analysis';
  cv.imshow(winName, resizedFrame);

  // Exit the loop if the 'q' key is pressed
  if (cv.waitKey(1000 / FPS) === 113) {
    break;
  }
}

// Release resources
cap.release();
cv.destroyAllWindows();
