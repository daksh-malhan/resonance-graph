const $ = (id) => document.getElementById(id);
const output = $("output");
let selectedVideoId = "";
let graphData = { nodes: [], links: [] };

function write(text) {
  output.textContent = `${new Date().toLocaleTimeString()}  ${text}\n\n${output.textContent}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function jsonBody(value) {
  return { method: "POST", body: JSON.stringify(value) };
}

function numberOrNull(value) {
  if (value === "" || value === null || value === undefined) return null;
  return Number(value);
}

async function refreshStatus() {
  const data = await api("/api/status");
  $("statusStrip").innerHTML = data.checks
    .map((check) => `
      <div class="status-pill ${check.ok ? "ok" : "fail"}">
        <strong>${check.ok ? "OK" : "FAIL"}</strong> ${check.label}
        <span>${check.message || ""}</span>
      </div>
    `)
    .join("");
}

async function setupDb() {
  write("Setting up Neo4j schema...");
  await api("/api/setup-db", jsonBody({}));
  write("Neo4j schema is ready.");
  await refreshGraph();
}

async function startJob(path, payload) {
  const data = await api(path, jsonBody(payload));
  write(`Started job ${data.job_id}`);
  pollJob(data.job_id);
}

async function pollJob(jobId) {
  const data = await api(`/api/jobs/${jobId}`);
  const job = data.job;
  write(`${job.kind}: ${job.state} - ${job.message}`);
  if (job.state === "queued" || job.state === "running") {
    if (["caption_ingesting", "writing_graph", "caption_ready", "local_transcribing", "merging_transcripts"].includes(job.stage)) {
      await refreshEpisodes();
      await refreshGraph();
    }
    window.setTimeout(() => pollJob(jobId), 3000);
    return;
  }
  if (job.error) write(job.error);
  if (job.result) write(JSON.stringify(job.result, null, 2));
  if (job.result?.local_transcription_job_id) {
    pollBackgroundJob(job.result.local_transcription_job_id);
  }
  await refreshEpisodes();
  await refreshGraph();
}

async function pollBackgroundJob(jobId) {
  const data = await api(`/api/background-jobs/${jobId}`);
  const job = data.job;
  write(`local merge: ${job.state} - ${job.message}`);
  if (job.state === "queued" || job.state === "running") {
    if (["writing_graph", "complete", "embedding_chunks", "merging_transcripts"].includes(job.stage)) {
      await refreshEpisodes();
      await refreshGraph();
    }
    window.setTimeout(() => pollBackgroundJob(jobId), 5000);
    return;
  }
  if (job.error) write(job.error);
  if (job.result) write(JSON.stringify(job.result, null, 2));
  await refreshEpisodes();
  await refreshGraph();
}

async function previewChannel() {
  const payload = channelPayload();
  write("Discovering channel videos...");
  const data = await api("/api/channel-preview", jsonBody(payload));
  write(
    `Preview returned ${data.videos.length} long-form video(s):\n` +
      data.videos
        .map((video, index) => `${index + 1}. ${video.video_id} | ${video.title}`)
        .join("\n")
  );
}

function channelPayload() {
  return {
    url: $("channelUrl").value.trim(),
    limit: numberOrNull($("channelLimit").value),
    min_duration: numberOrNull($("channelMinDuration").value),
    force: $("channelForce").checked,
  };
}

async function askQuestion() {
  const question = $("question").value.trim();
  if (!question) return write("Question is required.");
  write("Asking local RAG...");
  const data = await api(
    "/api/ask",
    jsonBody({
      question,
      top_k: numberOrNull($("topK").value),
      neighbors: numberOrNull($("neighbors").value),
      video_id: $("askEpisode").value || null,
    })
  );
  write(data.answer);
}

async function refreshEpisodes() {
  const data = await api("/api/episodes");
  const list = $("episodeList");
  const selector = $("graphEpisode");
  const askSelector = $("askEpisode");
  list.innerHTML = "";
  selector.innerHTML = '<option value="">All episodes</option>';
  askSelector.innerHTML = '<option value="">All episodes</option>';
  for (const episode of data.episodes) {
    const item = document.createElement("div");
    item.className = "episode";
    const transcriptStatus = episode.transcript_status || "unknown transcript";
    item.innerHTML = `
      <strong>${episode.title}</strong>
      <span>${episode.video_id} · ${episode.channel || "unknown"} · chunks ${episode.chunk_count} · ${transcriptStatus}</span>
    `;
    item.onclick = () => {
      selectedVideoId = episode.video_id;
      selector.value = episode.video_id;
      askSelector.value = episode.video_id;
      refreshGraph();
      inspectEpisode(episode.video_id);
    };
    list.appendChild(item);

    const option = document.createElement("option");
    option.value = episode.video_id;
    option.textContent = episode.title;
    selector.appendChild(option);

    const askOption = document.createElement("option");
    askOption.value = episode.video_id;
    askOption.textContent = episode.title;
    askSelector.appendChild(askOption);
  }
  if (selectedVideoId) {
    selector.value = selectedVideoId;
    askSelector.value = selectedVideoId;
  }
}

async function inspectEpisode(videoId) {
  const data = await api(`/api/episodes/${encodeURIComponent(videoId)}`);
  write(JSON.stringify(data.episode, null, 2));
}

async function refreshGraph() {
  const limit = Number($("graphLimit").value || 250);
  const selected = $("graphEpisode").value || selectedVideoId;
  const query = new URLSearchParams({ limit: String(limit) });
  if (selected) query.set("video_id", selected);
  graphData = await api(`/api/graph?${query.toString()}`);
  drawGraph(graphData);
}

function drawGraph(data) {
  const svg = $("graphSvg");
  const width = svg.clientWidth || 800;
  const height = svg.clientHeight || 420;
  svg.innerHTML = "";

  const nodes = data.nodes.map((node) => ({ ...node }));
  const links = data.links.map((link) => ({ ...link }));
  const byId = new Map(nodes.map((node) => [node.id, node]));

  const chunkNodes = nodes.filter((node) => node.label === "Chunk");
  for (const node of nodes) {
    if (node.label === "Source") {
      node.x = width * 0.14;
      node.y = height * 0.58;
    } else if (node.label === "Episode") {
      node.x = width * 0.46;
      node.y = height * 0.48;
    } else {
      const ordinal = Number(node.properties?.ordinal ?? chunkNodes.indexOf(node));
      const angle = ordinal * 2.399963 + 0.45;
      const radius = Math.min(width, height) * (0.18 + (ordinal % 9) * 0.025);
      node.x = width * 0.52 + Math.cos(angle) * radius * 1.52;
      node.y = height * 0.5 + Math.sin(angle) * radius * 0.82;
    }
    node.vx = 0;
    node.vy = 0;
  }

  const ns = "http://www.w3.org/2000/svg";
  const linkLayer = document.createElementNS(ns, "g");
  const nodeLayer = document.createElementNS(ns, "g");
  svg.append(linkLayer, nodeLayer);

  const linkEls = links.map((link) => {
    const path = document.createElementNS(ns, "path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", link.type === "HAS_EPISODE" ? "rgba(40, 109, 105, 0.42)" : "rgba(24, 23, 19, 0.14)");
    path.setAttribute("stroke-width", link.type === "HAS_EPISODE" ? "1.4" : "0.85");
    linkLayer.appendChild(path);
    return path;
  });

  const nodeEls = nodes.map((node) => {
    const group = document.createElementNS(ns, "g");
    group.style.cursor = "pointer";
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("r", node.label === "Episode" ? "9" : node.label === "Source" ? "7" : "3.4");
    circle.setAttribute("fill", node.label === "Episode" ? "#8a3340" : node.label === "Source" ? "#286d69" : "#8b7e66");
    circle.setAttribute("stroke", "rgba(24, 23, 19, 0.3)");
    circle.setAttribute("stroke-width", "0.7");
    const title = document.createElementNS(ns, "title");
    title.textContent = `${node.label}: ${node.title}`;
    group.append(circle, title);
    if (node.label !== "Chunk") {
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", "13");
      label.setAttribute("y", "4");
      label.setAttribute("fill", node.label === "Episode" ? "#181713" : "#286d69");
      label.setAttribute("font-size", "11");
      label.setAttribute("font-weight", "500");
      label.textContent = node.label;
      group.appendChild(label);
    }
    group.onclick = () => write(JSON.stringify(node.properties, null, 2));
    nodeLayer.appendChild(group);
    return group;
  });

  for (let tick = 0; tick < 220; tick++) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = a.x - b.x;
        let dy = a.y - b.y;
      let d2 = dx * dx + dy * dy || 1;
      const force = Math.min(38 / d2, 0.42);
        a.vx += dx * force;
        a.vy += dy * force;
        b.vx -= dx * force;
        b.vy -= dy * force;
      }
    }
    for (const link of links) {
      const a = byId.get(link.source);
      const b = byId.get(link.target);
      if (!a || !b) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const target = link.type === "HAS_EPISODE" ? 170 : 105;
      const force = (dist - target) * 0.008;
      a.vx += (dx / dist) * force;
      a.vy += (dy / dist) * force;
      b.vx -= (dx / dist) * force;
      b.vy -= (dy / dist) * force;
    }
    for (const node of nodes) {
      const anchorX = node.label === "Source" ? width * 0.14 : node.label === "Episode" ? width * 0.46 : width * 0.58;
      const anchorY = node.label === "Source" ? height * 0.58 : height * 0.48;
      node.vx += (anchorX - node.x) * 0.0015;
      node.vy += (anchorY - node.y) * 0.0012;
      node.x = Math.max(18, Math.min(width - 18, node.x + node.vx));
      node.y = Math.max(18, Math.min(height - 18, node.y + node.vy));
      node.vx *= 0.9;
      node.vy *= 0.9;
    }
  }

  links.forEach((link, index) => {
    const a = byId.get(link.source);
    const b = byId.get(link.target);
    if (!a || !b) return;
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const bend = link.type === "HAS_EPISODE" ? 0.08 : 0.045;
    const controlX = midX - dy * bend;
    const controlY = midY + dx * bend;
    linkEls[index].setAttribute("d", `M ${a.x} ${a.y} Q ${controlX} ${controlY} ${b.x} ${b.y}`);
  });
  nodes.forEach((node, index) => {
    nodeEls[index].setAttribute("transform", `translate(${node.x}, ${node.y})`);
  });
}

async function resetDb() {
  if (!window.confirm("Delete all graph data from Neo4j?")) return;
  await api("/api/reset-db", jsonBody({ confirm: true }));
  write("Graph data deleted.");
  await refreshEpisodes();
  await refreshGraph();
}

async function clearFiles() {
  if (!window.confirm("Delete local media, transcripts, chunks, and embeddings?")) return;
  await api("/api/clear-cache", jsonBody({ confirm: true, include_models: $("includeModels").checked }));
  write("Local files cleared.");
}

function bind() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      $(`tab-${tab.dataset.tab}`).classList.add("active");
    };
  });
  $("refreshStatus").onclick = refreshStatus;
  $("setupDb").onclick = setupDb;
  $("ingestVideo").onclick = () => startJob("/api/ingest-url", {
    url: $("videoUrl").value.trim(),
    force: $("videoForce").checked,
  });
  $("previewChannel").onclick = previewChannel;
  $("ingestChannel").onclick = () => startJob("/api/ingest-channel", channelPayload());
  $("askButton").onclick = askQuestion;
  $("refreshEpisodes").onclick = refreshEpisodes;
  $("refreshGraph").onclick = refreshGraph;
  $("graphEpisode").onchange = () => {
    selectedVideoId = $("graphEpisode").value;
    $("askEpisode").value = selectedVideoId;
    refreshGraph();
  };
  $("askEpisode").onchange = () => {
    selectedVideoId = $("askEpisode").value;
    $("graphEpisode").value = selectedVideoId;
    refreshGraph();
  };
  $("graphLimit").onchange = refreshGraph;
  $("resetDb").onclick = resetDb;
  $("clearFiles").onclick = clearFiles;
  $("clearOutput").onclick = () => {
    output.textContent = "";
  };
}

bind();
refreshStatus().catch((error) => write(error.message));
refreshEpisodes().then(refreshGraph).catch((error) => write(error.message));
