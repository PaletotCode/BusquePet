const apiBaseInput = document.getElementById("api-base");
const wsUrlInput = document.getElementById("ws-url");
const connectBtn = document.getElementById("connect-ws");
const disconnectBtn = document.getElementById("disconnect-ws");
const wsStatus = document.getElementById("ws-status");

const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const uploadBtn = document.getElementById("upload-btn");
const resetBtn = document.getElementById("reset-btn");
const fileInfo = document.getElementById("file-info");
const uploadStatus = document.getElementById("upload-status");

const jobIdInput = document.getElementById("job-id");
const latencyInput = document.getElementById("latency");
const refreshBtn = document.getElementById("refresh-job");
const statusFeed = document.getElementById("status-feed");
const resultsGrid = document.getElementById("results-grid");
const resultSummary = document.getElementById("result-summary");
const logOutput = document.getElementById("log-output");
const preview = document.getElementById("preview");

let selectedFile = null;
let websocket = null;
let currentJobId = null;
let jobStartTime = null;

const log = (message, level = "info") => {
  const line = document.createElement("div");
  line.className = `log-line ${level}`;
  line.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
  logOutput.prepend(line);
};

const appendStatus = (text) => {
  const li = document.createElement("li");
  li.textContent = text;
  statusFeed.prepend(li);
};

const renderResults = (result) => {
  if (!resultsGrid) return;
  const { matches = [] } = result || {};
  resultsGrid.innerHTML = "";
  if (!matches.length) {
    resultsGrid.innerHTML = `<div class="placeholder">Sem matches ainda.</div>`;
    if (resultSummary) resultSummary.textContent = "Aguarde os resultados.";
    return;
  }
  if (resultSummary) resultSummary.textContent = `Top ${matches.length} matches`;
  matches.forEach((match, index) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const score = (match.score ?? 0) * 100;
    const apiBase = apiBaseInput.value.trim().replace(/\/$/, "");
    const rawPath = match.image_path || "";
    const normalizedPath = rawPath.replace(/^\/+/, "");
    const imgSrc = rawPath ? `${apiBase}/files/${normalizedPath}` : "";
    card.innerHTML = `
      <div class="result-meta">
        <strong>#${index + 1} ${match.breed ?? "-"}</strong>
        <span class="score">${score.toFixed(1)}%</span>
      </div>
      <div class="muted">${match.image_id ?? "-"}</div>
      ${
        imgSrc
          ? `<img src="${imgSrc}" alt="preview" onerror="this.style.display='none'">`
        : `<div class="placeholder">Sem preview disponível</div>`
      }
      <div class="muted">${rawPath}</div>
    `;
    resultsGrid.appendChild(card);
  });
};

const resetUI = () => {
  selectedFile = null;
  currentJobId = null;
  jobStartTime = null;
  fileInfo.textContent = "";
  uploadStatus.textContent = "";
  jobIdInput.value = "";
  latencyInput.value = "";
  statusFeed.innerHTML = "";
  renderResults();
  uploadBtn.disabled = true;
};

const connectWebSocket = () => {
  const url = wsUrlInput.value.trim();
  if (!url) return alert("Informe a URL do WebSocket.");

  if (websocket) {
    websocket.close();
  }

  websocket = new WebSocket(url);
  websocket.onopen = () => {
    wsStatus.textContent = "conectado";
    wsStatus.style.color = "#16a34a";
    log("WebSocket conectado");
  };
  websocket.onerror = (err) => {
    log(`Erro no WebSocket: ${err.message ?? err}`, "error");
  };
  websocket.onclose = () => {
    wsStatus.textContent = "desconectado";
    wsStatus.style.color = "#dc2626";
    log("WebSocket desconectado", "warn");
  };
  websocket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (!payload?.job_id) return;
      appendStatus(`[WS] ${payload.job_id}: ${payload.status ?? payload.type}`);
      if (payload.job_id === currentJobId && payload.type === "job.result") {
        renderResults(payload.result);
        if (jobStartTime) {
          latencyInput.value = Date.now() - jobStartTime;
        }
      }
    } catch (error) {
      log(`Falha ao processar evento do WS: ${error}`, "error");
    }
  };
};

const fetchJob = async (jobId) => {
  if (!jobId) return;
  const endpoint = `${apiBaseInput.value.trim()}/v1/jobs/${jobId}`;
  const response = await fetch(endpoint);
  if (!response.ok) throw new Error(`Erro ao consultar job: ${response.status}`);
  return response.json();
};

const handleUpload = async () => {
  if (!selectedFile) return alert("Escolha uma imagem primeiro.");
  const apiBase = apiBaseInput.value.trim();
  if (!apiBase) return alert("Informe a URL da API.");

  const form = new FormData();
  form.append("image", selectedFile);
  try {
    uploadBtn.disabled = true;
    uploadStatus.textContent = "Enviando...";
    jobStartTime = Date.now();
    const resp = await fetch(`${apiBase}/v1/ingest`, {
      method: "POST",
      body: form,
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.error ?? `status ${resp.status}`);
    }
    const data = await resp.json();
    currentJobId = data.job_id;
    jobIdInput.value = currentJobId;
    uploadStatus.textContent = `Job ${currentJobId} enfileirado.`;
    appendStatus(`[HTTP] ${currentJobId}: pending`);
  } catch (error) {
    log(`Upload falhou: ${error.message}`, "error");
    uploadStatus.textContent = `Erro: ${error.message}`;
  } finally {
    uploadBtn.disabled = !selectedFile;
  }
};

const handleRefreshJob = async () => {
  if (!currentJobId) return alert("Envie uma imagem primeiro.");
  try {
    const payload = await fetchJob(currentJobId);
    appendStatus(`[REST] ${payload.job.status}`);
    if (payload.result) {
      renderResults(payload.result);
      if (jobStartTime) latencyInput.value = Date.now() - jobStartTime;
    }
  } catch (error) {
    log(error.message, "error");
  }
};

const handleFiles = (files) => {
  if (!files?.length) return;
  const file = files[0];
  selectedFile = file;
  fileInfo.textContent = `${file.name} — ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  uploadBtn.disabled = false;
  uploadStatus.textContent = "";
  preview.innerHTML = "";
  const url = URL.createObjectURL(file);
  const img = document.createElement("img");
  img.src = url;
  img.alt = "Preview";
  preview.appendChild(img);
};

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => handleFiles(event.target.files));

["dragenter", "dragover"].forEach((eventName) => {
  dropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropArea.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropArea.classList.remove("dragover");
  });
});

dropArea.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) {
    handleFiles(event.dataTransfer.files);
  }
});

uploadBtn.addEventListener("click", handleUpload);
resetBtn.addEventListener("click", resetUI);
refreshBtn.addEventListener("click", handleRefreshJob);
connectBtn.addEventListener("click", connectWebSocket);
disconnectBtn.addEventListener("click", () => websocket?.close());

// Inicializa UI vazia
renderResults();
log("Console carregado. Configure os endpoints e conecte o WebSocket.");
