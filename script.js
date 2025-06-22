// DOM Elements
const webcamVideo = document.getElementById("webcam");
const faceCanvas = document.getElementById("face-canvas");
const videoContainer = document.getElementById("video-container");
const toggleWebcamBtn = document.getElementById("toggle-webcam");
const toggleRecognitionBtn = document.getElementById("toggle-recognition");
const captureFaceBtn = document.getElementById("capture-face");
const newFaceLabel = document.getElementById("new-face-label");
const facesList = document.getElementById("faces-list");
const statusLog = document.getElementById("status-log");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingStatus = document.getElementById("loading-status");
const statusIndicator = document.getElementById("status-indicator");
const fpsCounter = document.getElementById("fps-counter");

// App State
let isWebcamActive = false;
let isRecognitionActive = true;
let webcamStream = null;
let recognitionInterval = null;
let labeledFaceDescriptors = [];
let faceMatcher = null;
let lastFrameTime = 0;
let frameCount = 0;
let fps = 0;

// Initialize the app
async function init() {
  try {
    updateLoadingStatus("Loading face detection models...");
    await loadModels();

    updateLoadingStatus("Starting webcam...");
    await startWebcam();

    // Load faces from localStorage
    loadFacesFromStorage();

    // Hide loading overlay
    loadingOverlay.style.display = "none";

    // Start recognition
    startFaceRecognition();

    // Update FPS counter
    setInterval(updateFPS, 1000);

    // Log status
    logStatus("System initialized and ready");
    logStatus("Face recognition active");

    // Setup event listeners
    setupEventListeners();
  } catch (error) {
    console.error("Initialization error:", error);
    logStatus("ERROR: " + error.message, true);
    updateLoadingStatus("Error: " + error.message);
  }
}

// Load face-api.js models
async function loadModels() {
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(
        "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"
      ),
      faceapi.nets.faceLandmark68Net.loadFromUri(
        "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"
      ),
      faceapi.nets.faceRecognitionNet.loadFromUri(
        "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"
      ),
      faceapi.nets.ssdMobilenetv1.loadFromUri(
        "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"
      ),
    ]);
    logStatus("Neural network models loaded");
  } catch (error) {
    throw new Error("Failed to load face detection models: " + error.message);
  }
}

// Start webcam
async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });

    webcamVideo.srcObject = webcamStream;

    await new Promise((resolve) => {
      webcamVideo.onloadedmetadata = () => {
        // Set canvas dimensions to match video
        faceCanvas.width = webcamVideo.videoWidth;
        faceCanvas.height = webcamVideo.videoHeight;

        // Ensure video container has correct dimensions
        videoContainer.style.width = webcamVideo.videoWidth + "px";
        videoContainer.style.height = webcamVideo.videoHeight + "px";

        isWebcamActive = true;
        resolve();
      };
    });

    logStatus("Camera initialized");
  } catch (error) {
    throw new Error("Failed to access webcam: " + error.message);
  }
}

// Stop webcam
function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamVideo.srcObject = null;
    isWebcamActive = false;

    // Clear canvas
    const ctx = faceCanvas.getContext("2d");
    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);

    logStatus("Camera stopped");
  }
}

// Toggle webcam
async function toggleWebcam() {
  if (isWebcamActive) {
    stopWebcam();
    toggleWebcamBtn.textContent = "START CAMERA";
    statusIndicator.classList.remove("status-active");
    statusIndicator.classList.add("status-inactive");
  } else {
    await startWebcam();
    toggleWebcamBtn.textContent = "STOP CAMERA";
    statusIndicator.classList.remove("status-inactive");
    statusIndicator.classList.add("status-active");

    // Restart recognition if it was active
    if (isRecognitionActive) {
      startFaceRecognition();
    }
  }
}

// Toggle face recognition
function toggleRecognition() {
  if (isRecognitionActive) {
    stopFaceRecognition();
    toggleRecognitionBtn.textContent = "RESUME RECOGNITION";
    logStatus("Face recognition paused");
  } else {
    startFaceRecognition();
    toggleRecognitionBtn.textContent = "PAUSE RECOGNITION";
    logStatus("Face recognition resumed");
  }
}

// Start face recognition loop
function startFaceRecognition() {
  if (!isWebcamActive || recognitionInterval) return;

  isRecognitionActive = true;
  recognitionInterval = setInterval(recognizeFaces, 100);
}

// Stop face recognition loop
function stopFaceRecognition() {
  if (recognitionInterval) {
    clearInterval(recognitionInterval);
    recognitionInterval = null;
    isRecognitionActive = false;

    // Clear canvas
    const ctx = faceCanvas.getContext("2d");
    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  }
}

// Recognize faces in current video frame
async function recognizeFaces() {
  if (!isWebcamActive || webcamVideo.readyState !== 4) return;
  // Track frame time for FPS calculation
  const now = performance.now();
  frameCount++;

  try {
    // Detect all faces with landmarks and descriptors
    const detections = await faceapi
      .detectAllFaces(
        webcamVideo,
        new faceapi.TinyFaceDetectorOptions({ inputSize: 320 })
      )
      .withFaceLandmarks()
      .withFaceDescriptors();

    // Clear previous drawings
    const ctx = faceCanvas.getContext("2d");
    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);

    // Match and draw each face
    if (detections.length > 0) {
      detections.forEach((detection) => {
        const box = detection.detection.box;

        // Draw face box
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#00ff00";
        ctx.beginPath();
        ctx.rect(box.x, box.y, box.width, box.height);
        ctx.stroke();

        // Match face if we have reference faces
        let label = "Unknown";
        if (faceMatcher && detection.descriptor) {
          const match = faceMatcher.findBestMatch(detection.descriptor);
          label = match.label;

          // Draw confidence if not unknown
          if (label !== "unknown") {
            const confidence = Math.round(match.distance * 100);
            label = `${label} (${confidence}%)`;
          }
        }

        // Draw label background
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(box.x, box.y - 30, box.width, 30);

        // Draw label text
        ctx.fillStyle = label === "Unknown" ? "#ff0000" : "#00ff00";
        ctx.font = '16px "Space Mono"';
        ctx.fillText(label, box.x + 5, box.y - 10);
      });
    }
  } catch (error) {
    console.error("Recognition error:", error);
  }
}

// Capture current face
async function captureFace() {
  if (!isWebcamActive) {
    logStatus("ERROR: Camera must be active to capture face", true);
    return;
  }

  const label = newFaceLabel.value.trim();
  if (!label) {
    logStatus("ERROR: Please enter a name for this face", true);
    return;
  }

  try {
    // Temporarily pause recognition
    const wasRecognitionActive = isRecognitionActive;
    if (wasRecognitionActive) {
      stopFaceRecognition();
    }

    logStatus(`Capturing face for: ${label}...`);

    // Detect face with highest confidence
    const detections = await faceapi
      .detectAllFaces(
        webcamVideo,
        new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })
      )
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (detections.length === 0) {
      logStatus("ERROR: No face detected. Please try again.", true);

      // Resume recognition if it was active
      if (wasRecognitionActive) {
        startFaceRecognition();
      }
      return;
    }

    if (detections.length > 1) {
      logStatus(
        "WARNING: Multiple faces detected. Using the most prominent face.",
        true
      );
    }

    // Use the first (most confident) face
    const faceDescriptor = detections[0].descriptor;

    // Check if this label already exists
    const existingIndex = labeledFaceDescriptors.findIndex(
      (lfd) => lfd.label === label
    );
    if (existingIndex !== -1) {
      // Update existing face descriptor
      labeledFaceDescriptors[existingIndex] =
        new faceapi.LabeledFaceDescriptors(label, [faceDescriptor]);
      logStatus(`Updated face data for: ${label}`);
    } else {
      // Add new face descriptor
      const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(label, [
        faceDescriptor,
      ]);
      labeledFaceDescriptors.push(labeledFaceDescriptor);
      logStatus(`Added new face: ${label}`);
    }

    // Update face matcher
    updateFaceMatcher();

    // Update UI
    updateFacesList();
    newFaceLabel.value = "";

    // Show face preview
    const previewDiv = document.getElementById("face-preview");
    const box = detections[0].detection.box;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = box.width;
    tempCanvas.height = box.height;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(
      webcamVideo,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      box.width,
      box.height
    );
    previewDiv.innerHTML = ""; // Clear previous
    previewDiv.appendChild(tempCanvas);

    // Resume recognition if it was active
    if (wasRecognitionActive) {
      startFaceRecognition();
    }
  } catch (error) {
    console.error("Face capture error:", error);
    logStatus("ERROR: Failed to capture face: " + error.message, true);
  }
}

// Update face matcher with current labeled descriptors
function updateFaceMatcher() {
  if (labeledFaceDescriptors.length > 0) {
    faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  }
}

// Update the list of stored faces in the UI
function updateFacesList() {
  facesList.innerHTML = "";

  if (labeledFaceDescriptors.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No stored identities";
    facesList.appendChild(li);
    // Save to localStorage
    saveFacesToStorage();
    return;
  }

  labeledFaceDescriptors.forEach((descriptor) => {
    const li = document.createElement("li");
    li.className = "flex justify-between items-center";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = descriptor.label;

    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "×";
    deleteBtn.className = "text-red-500 hover:text-red-300 font-bold";
    deleteBtn.onclick = () => deleteFace(descriptor.label);

    li.appendChild(nameSpan);
    li.appendChild(deleteBtn);
    facesList.appendChild(li);
  });

  // Save to localStorage
  saveFacesToStorage();
}

function deleteFace(label) {
  labeledFaceDescriptors = labeledFaceDescriptors.filter(
    (lfd) => lfd.label !== label
  );
  updateFaceMatcher();
  updateFacesList();
  logStatus(`Removed face: ${label}`);
}

// Load sample faces (for demo purposes)


// Setup event listeners
function setupEventListeners() {
  toggleWebcamBtn.addEventListener("click", toggleWebcam);
  toggleRecognitionBtn.addEventListener("click", toggleRecognition);
  captureFaceBtn.addEventListener("click", captureFace);

  // Handle window resize
  window.addEventListener("resize", () => {
    if (isWebcamActive) {
      // Ensure canvas stays aligned with video
      faceCanvas.width = webcamVideo.videoWidth;
      faceCanvas.height = webcamVideo.videoHeight;
    }
  });
}

// Log status message
function logStatus(message, isError = false) {
  const timestamp = new Date().toLocaleTimeString();
  const logEntry = document.createElement("div");
  logEntry.className = isError ? "text-red-500" : "";
  logEntry.textContent = `[${timestamp}] ${message}`;

  statusLog.prepend(logEntry);

  // Limit log entries
  if (statusLog.children.length > 20) {
    statusLog.removeChild(statusLog.lastChild);
  }
}

// Update loading status
function updateLoadingStatus(message) {
  loadingStatus.textContent = message;
}

// Update FPS counter
function updateFPS() {
  if (isWebcamActive && isRecognitionActive) {
    const now = performance.now();
    fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
    lastFrameTime = now;
    frameCount = 0;

    fpsCounter.textContent = `FPS: ${fps}`;
  } else {
    fpsCounter.textContent = "FPS: --";
  }
}

// Save faces to localStorage
function saveFacesToStorage() {
  // Convert descriptors to plain arrays for storage
  const data = labeledFaceDescriptors.map((lfd) => ({
    label: lfd.label,
    descriptors: lfd.descriptors.map((desc) => Array.from(desc)),
  }));
  localStorage.setItem("faces", JSON.stringify(data));
}

// Load faces from localStorage
function loadFacesFromStorage() {
  const data = JSON.parse(localStorage.getItem("faces") || "[]");
  labeledFaceDescriptors = data.map((face) =>
    new faceapi.LabeledFaceDescriptors(
      face.label,
      face.descriptors.map((desc) => new Float32Array(desc))
    )
  );
  updateFaceMatcher();
  updateFacesList();
}

// Start the app
init();
