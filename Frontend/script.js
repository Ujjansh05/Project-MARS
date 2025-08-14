const landingPage = document.getElementById('landingPage');
const controlCenter = document.getElementById('controlCenter');
const startButton = document.getElementById('startButton');

startButton.addEventListener('click', () => {
    landingPage.classList.add('hidden');
    controlCenter.classList.remove('hidden');
    setTimeout(() => {
        controlCenter.style.opacity = 1;
    }, 10);
});

let uploadedFile = null;
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultsPanel = document.getElementById("resultsPanel");
const mapPanel = document.getElementById("mapPanel");

fileInput.addEventListener("change", (event) => {
  uploadedFile = event.target.files[0];
  if (uploadedFile) {
    analyzeBtn.disabled = false;
    resultsPanel.innerHTML = `
      <div class="vessel">
        <span>${uploadedFile.name}</span><span class="badge ready">Ready</span>
      </div>
    `;
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!uploadedFile) return;

  const formData = new FormData();
  formData.append("file", uploadedFile);

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
  mapPanel.innerHTML = '<i class="fas fa-spinner fa-spin" style="font-size: 3rem;"></i><p style="margin-top: 1rem;">Running model, please wait...</p>';

  try {
    const res = await fetch("http://127.0.0.1:5000/upload", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

  if (res.ok) {
  mapPanel.innerHTML = `
    <video width="100%" controls autoplay loop>
      <source src="${data.video_url}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <a href="${data.csv_url}" class="download-link" download>
      <i class="fas fa-download"></i> Download CSV Log
    </a>
  `;

  // Auto-download processed video
  const downloadLink = document.createElement("a");
  downloadLink.href = data.video_url;
  downloadLink.download = "processed_video.mp4"; // Auto-save name
  document.body.appendChild(downloadLink);
  downloadLink.click();
  document.body.removeChild(downloadLink);

  resultsPanel.innerHTML += `
    <div class="vessel">
        <span>Tracking Complete</span><span class="badge done">Done</span>
    </div>
  `;
}

else {
      mapPanel.innerHTML = `<p style="color: #e57373;">Analysis Failed</p>`;
      resultsPanel.innerHTML += `
        <div class="vessel">
            <span>Error: ${data.error}</span><span class="badge error">Failed</span>
        </div>
      `;
      alert("Error: " + data.error);
    }
  } catch (err) {
    console.error(err);
    mapPanel.innerHTML = `<p style="color: #e57373;">Connection Error</p>`;
    resultsPanel.innerHTML += `
        <div class="vessel">
            <span>Server Unreachable</span><span class="badge error">Failed</span>
        </div>
      `;
    alert("Failed to connect to the analysis server.");
  }

  analyzeBtn.disabled = false;
  analyzeBtn.innerHTML = '<i class="fas fa-cogs"></i> Analyze';
});
