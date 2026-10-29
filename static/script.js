
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const placeholder = document.getElementById("cameraPlaceholder");
const cameraStatus = document.getElementById("cameraStatus");
const message = document.getElementById("message");

const emotionEmoji = document.getElementById("emotionEmoji");
const emotionName = document.getElementById("emotionName");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");
const faceCount = document.getElementById("faceCount");
const analysisStatus = document.getElementById("analysisStatus");

const canvas = document.createElement("canvas");
const canvasCtx = canvas.getContext("2d");

const emotionEmojis = {
    angry: "😠",
    disgust: "🤢",
    fear: "😨",
    happy: "😄",
    neutral: "😐",
    sad: "😢",
    surprise: "😲"
};

let stream = null;
let running = false;
let busy = false;
let timer = null;
let lastRequestId = 0;

function resetResults() {
    emotionEmoji.textContent = "😐";
    emotionName.textContent = "Waiting...";
    confidenceText.textContent = "No prediction yet";
    confidenceValue.textContent = "—";
    confidenceBar.style.width = "0%";
    faceCount.textContent = "0";
    analysisStatus.textContent = "Idle";
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function stopCamera() {
    running = false;
    lastRequestId++;

    if (timer !== null) {
        clearTimeout(timer);
        timer = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    busy = false;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    placeholder.classList.remove("hidden");
    cameraStatus.textContent = "Camera off";
    cameraStatus.classList.remove("active");

    startBtn.disabled = false;
    stopBtn.disabled = true;

    analysisStatus.textContent = "Idle";
    message.textContent = "Camera stopped.";

    resetResults();
}

async function startCamera() {
    if (running) return;

    if (!navigator.mediaDevices?.getUserMedia) {
        message.textContent =
            "Camera access requires HTTPS or localhost in a supported browser.";
        return;
    }

    startBtn.disabled = true;
    message.textContent = "Requesting camera permission...";

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        video.srcObject = stream;
        await video.play();

        running = true;
        placeholder.classList.add("hidden");

        cameraStatus.textContent = "Camera active";
        cameraStatus.classList.add("active");

        startBtn.disabled = true;
        stopBtn.disabled = false;

        message.textContent = "Camera started. Detecting faces...";
        analysisStatus.textContent = "Starting";

        // Match canvas coordinates to the actual video resolution.
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        resetResults();
        schedulePrediction();

    } catch (error) {
        console.error(error);

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        video.srcObject = null;
        running = false;
        startBtn.disabled = false;

        message.textContent =
            "Could not start the camera. Check browser permissions and try again.";
    }
}

function schedulePrediction() {
    if (!running) return;

    // Limit requests to avoid overwhelming the API.
    timer = setTimeout(async () => {
        await predictFrame();
        schedulePrediction();
    }, 400);
}

async function predictFrame() {
    if (!running || busy || video.readyState < 2) return;

    busy = true;
    const requestId = lastRequestId;

    try {
        canvasCtx.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, "image/jpeg", 0.75);
        });

        if (!blob || !running || requestId !== lastRequestId) {
            return;
        }

        const formData = new FormData();
        formData.append("file", blob, "webcam.jpg");

        analysisStatus.textContent = "Analyzing";

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            let detail = "Prediction request failed.";

            try {
                const errorData = await response.json();
                detail = errorData.detail || detail;
            } catch (_) {
                // Keep the default error message.
            }

            throw new Error(detail);
        }

        const data = await response.json();

        if (!running || requestId !== lastRequestId) return;

        displayResults(data);

    } catch (error) {
        console.error("Prediction error:", error);

        if (running && requestId === lastRequestId) {
            analysisStatus.textContent = "Error";
            message.textContent =
                "Could not get a prediction. Check that FastAPI is running.";
        }
    } finally {
        busy = false;
    }
}

function displayResults(data) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const predictions = data.predictions || [];
    faceCount.textContent = data.faces_detected ?? predictions.length;

    if (predictions.length === 0) {
        emotionEmoji.textContent = "🔎";
        emotionName.textContent = "No face found";
        confidenceText.textContent = "Position your face in view";
        confidenceValue.textContent = "—";
        confidenceBar.style.width = "0%";
        analysisStatus.textContent = "Searching";
        message.textContent = "No face detected. Adjust your position.";
        return;
    }

    // Show the first detected face in the summary.
    const result = predictions[0];
    const emotion = String(result.emotion || "unknown").toLowerCase();

    // The model score is displayed as a percentage.
    const confidence = Math.max(
        0,
        Math.min(100, Number(result.confidence) || 0)
    );

    emotionEmoji.textContent = emotionEmojis[emotion] || "🙂";
    emotionName.textContent =
        emotion.charAt(0).toUpperCase() + emotion.slice(1);

    confidenceText.textContent = "Model prediction score";
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceBar.style.width = `${confidence}%`;

    analysisStatus.textContent = "Active";
    message.textContent =
        `${predictions.length} face(s) detected.`;

    // Draw bounding boxes for all detected faces.
    ctx.lineWidth = Math.max(2, overlay.width / 320);
    ctx.font = `bold ${Math.max(14, overlay.width / 35)}px Arial`;
    ctx.textBaseline = "bottom";

    predictions.forEach(item => {
        const box = item.box;

        if (!box) return;

        // The webcam video is mirrored in CSS, so mirror box positions too.
        const x = overlay.width - box.x - box.width;
        const y = box.y;
        const w = box.width;
        const h = box.height;

        ctx.strokeStyle = "#43d9e8";
        ctx.fillStyle = "#43d9e8";
        ctx.strokeRect(x, y, w, h);

        const label =
            `${item.emotion} ${Number(item.confidence).toFixed(1)}%`;

        const textWidth = ctx.measureText(label).width;
        const labelHeight = Math.max(24, overlay.width / 25);
        const labelY = Math.max(labelHeight, y);

        ctx.fillRect(
            x,
            labelY - labelHeight,
            textWidth + 12,
            labelHeight
        );

        ctx.fillStyle = "#090d18";
        ctx.fillText(label, x + 6, labelY - 4);
    });
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("pagehide", () => {
    running = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
