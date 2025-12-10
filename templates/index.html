<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Drowsiness Detector (EAR - Backend Flask)</title>
  <style>
    body {
      background: #000;
      color: #fff;
      text-align: center;
      font-family: Arial, sans-serif;
    }
    h1 {
      margin-top: 20px;
    }
    #video {
      display: none;
    }
    #canvas {
      width: 90%;
      max-width: 420px;
      border-radius: 12px;
      margin-top: 10px;
      background: #111;
    }
    button {
      padding: 10px 20px;
      margin-top: 10px;
      cursor: pointer;
    }
    #status {
      font-size: 22px;
      margin-top: 15px;
    }
    #earValue, #timerValue {
      font-size: 18px;
      margin-top: 5px;
    }
  </style>
</head>

<body>
  <h1>Drowsiness Detector (EAR)</h1>

  <button id="toggleAlarm">Enable Alarm</button>

  <br />
  <video id="video" playsinline></video>
  <canvas id="canvas"></canvas>

  <h2 id="status">Status: Loading...</h2>
  <div id="earValue">EAR: --</div>
  <div id="timerValue">Note: Drowsy if EAR &lt; 0.20 for 5 seconds</div>

  <!-- Alarm sound -->
  <audio id="alarmSound" src="/static/alarm.wav" preload="auto" loop></audio>

  <script>
    const videoEl = document.getElementById('video');
    const canvasEl = document.getElementById('canvas');
    const ctx = canvasEl.getContext('2d');

    const statusEl = document.getElementById('status');
    const earEl = document.getElementById('earValue');
    const timerEl = document.getElementById('timerValue');
    const toggleBtn = document.getElementById('toggleAlarm');
    const alarm = document.getElementById('alarmSound');

    let alarmEnabled = false;

    // Toggle alarm ON/OFF
    toggleBtn.addEventListener('click', async () => {
      try {
        // unlock audio on first interaction
        await alarm.play();
        alarm.pause();
        alarm.currentTime = 0;
      } catch (e) {
        console.log("Audio unlock:", e);
      }

      alarmEnabled = !alarmEnabled;
      if (alarmEnabled) {
        toggleBtn.textContent = "Disable Alarm";
      } else {
        toggleBtn.textContent = "Enable Alarm";
        alarm.pause();
        alarm.currentTime = 0;
      }
    });

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        videoEl.srcObject = stream;
        videoEl.play();
      })
      .catch(err => {
        console.error(err);
        statusEl.innerHTML = 'Status: <span style="color:yellow">Camera blocked</span>';
      });

    // Send frame to backend every 300ms
    setInterval(() => {
      if (!videoEl.videoWidth) return;

      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
      ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

      const frame = canvasEl.toDataURL("image/jpeg");

      fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frame })
      })
      .then(res => res.json())
      .then(data => {
        const status = data.status;
        const ear = data.ear;

        let color = "lime";
        if (status === "Drowsy") color = "red";
        else if (status === "No face") color = "yellow";

        statusEl.innerHTML = `Status: <span style="color:${color}">${status}</span>`;
        earEl.textContent = `EAR: ${ear.toFixed ? ear.toFixed(3) : ear}`;

        // Alarm only if enabled & drowsy
        if (alarmEnabled && status === "Drowsy") {
          if (alarm.paused) alarm.play();
        } else {
          alarm.pause();
          alarm.currentTime = 0;
        }
      })
      .catch(err => console.error("Predict error:", err));
    }, 300); // Must match FRAME_INTERVAL_SEC in Python
  </script>
</body>
</html>
