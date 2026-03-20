(function () {
  const API_BASE = "";

  const videoListEl = document.getElementById("videoList");
  const emptyStateEl = document.getElementById("emptyState");
  const panelTitleEl = document.getElementById("panelTitle");
  const messagesEl = document.getElementById("messages");
  const welcomeEl = document.getElementById("welcome");
  const userInputEl = document.getElementById("userInput");
  const chatForm = document.getElementById("chatForm");
  const btnTransmutar = document.getElementById("btnTransmutar");
  const modalOverlay = document.getElementById("modalOverlay");
  const processarForm = document.getElementById("processarForm");
  const btnCancelar = document.getElementById("btnCancelar");
  const loadingOverlay = document.getElementById("loadingOverlay");

  let videos = [];
  let selectedVideoId = null;
  const historyByVideo = {};

  function getHistory() {
    if (!selectedVideoId) return [];
    if (!historyByVideo[selectedVideoId]) historyByVideo[selectedVideoId] = [];
    return historyByVideo[selectedVideoId];
  }

  function addToHistory(role, content) {
    const h = getHistory();
    h.push({ role, content });
  }

  function renderVideos() {
    const empty = !videos.length;
    videoListEl.classList.toggle("empty", empty);
    if (empty) {
      videoListEl.innerHTML = "";
      emptyStateEl.style.display = "block";
      return;
    }
    emptyStateEl.style.display = "none";
    videoListEl.innerHTML = videos
      .map(
        (v) =>
          `<li><button type="button" class="video-item ${v.id === selectedVideoId ? "active" : ""}" data-video-id="${escapeAttr(v.id)}">
            <span class="video-name">${escapeHtml(v.nome)}</span>
          </button></li>`
      )
      .join("");

    videoListEl.querySelectorAll(".video-item").forEach((btn) => {
      btn.addEventListener("click", () => selectVideo(btn.dataset.videoId));
    });
  }

  function escapeAttr(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function selectVideo(videoId) {
    selectedVideoId = videoId;
    panelTitleEl.textContent = videos.find((v) => v.id === videoId)?.nome ?? videoId;
    userInputEl.disabled = false;
    userInputEl.focus();
    welcomeEl.style.display = getHistory().length ? "none" : "block";
    renderMessages();
    renderVideos();
  }

  function renderMessages() {
    const history = getHistory();
    const toShow = history.map((m) => m);
    const welcomeVisible = !toShow.length;
    welcomeEl.style.display = welcomeVisible ? "block" : "none";

    const existing = messagesEl.querySelectorAll(".msg");
    existing.forEach((n) => n.remove());

    toShow.forEach((m) => {
      const div = document.createElement("div");
      div.className = "msg " + m.role;
      div.innerHTML =
        '<span class="role">' +
        (m.role === "user" ? "Você" : "Cronista") +
        "</span><div class='content'>" +
        escapeHtml(m.content) +
        "</div>";
      messagesEl.appendChild(div);
    });

    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function fetchVideos() {
    try {
      const r = await fetch(API_BASE + "/videos");
      if (!r.ok) throw new Error("Falha ao carregar vídeos");
      const data = await r.json();
      videos = data.videos || [];
      renderVideos();
      if (selectedVideoId && !videos.some((v) => v.id === selectedVideoId)) {
        selectedVideoId = null;
        panelTitleEl.textContent = "Selecione um grimório";
        userInputEl.disabled = true;
        renderMessages();
      }
    } catch (e) {
      console.error(e);
      videos = [];
      renderVideos();
    }
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = userInputEl.value.trim();
    if (!text || !selectedVideoId) return;

    userInputEl.value = "";
    addToHistory("user", text);
    renderMessages();

    const botEntry = { role: "bot", content: "…" };
    addToHistory("bot", botEntry.content);
    renderMessages();

    const submitBtn = chatForm.querySelector('button[type="submit"]');
    if (submitBtn) submitBtn.disabled = true;
    userInputEl.disabled = true;

    let accumulated = "";
    let sources = [];
    const last = getHistory();
    const botIdx = last.findIndex((m) => m.role === "bot" && m.content === "…");
    const botContentEl = messagesEl.querySelector(".msg.bot:last-child .content");

    const setBotContent = (txt) => {
        if (botIdx !== -1) last[botIdx].content = txt;
        else if (last.length) last[last.length - 1].content = txt;
        if (botContentEl) botContentEl.textContent = txt;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      };

    try {
      const res = await fetch(API_BASE + "/perguntar/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: selectedVideoId, pergunta: text }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const detail = data.detail;
        const msg = Array.isArray(detail)
          ? (detail[0] && detail[0].msg) || JSON.stringify(detail)
          : typeof detail === "string"
            ? detail
            : "Erro ao consultar.";
        setBotContent("Erro: " + msg);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamDone = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === "sources" && data.sources) sources = data.sources;
              else if (data.type === "token" && data.content) {
                accumulated += data.content;
                setBotContent(accumulated);
              } else if (data.type === "agent_done") {
                const finalAnswer = data.answer != null ? String(data.answer) : accumulated;
                const srcs = data.sources || sources;
                const finalContent = srcs.length
                  ? finalAnswer + "\n\nFontes: " + srcs.map((s) => s.video_id || s.source).join(", ")
                  : finalAnswer;
                setBotContent(finalContent);
                if (botIdx !== -1) last[botIdx].content = finalContent;
                streamDone = true;
                break;
              } else if (data.type === "done") {
                // RAG stream finished, keep reading for agent_done
              } else if (data.type === "error") {
                setBotContent("Erro: " + (data.error || "Erro desconhecido."));
                streamDone = true;
                break;
              }
            } catch (_) {}
          }
        }
        if (streamDone) break;
      }
      if (!streamDone && accumulated && sources.length) {
        const suffix = "\n\nFontes: " + sources.map((s) => s.video_id).join(", ");
        setBotContent(accumulated + suffix);
        if (botIdx !== -1) last[botIdx].content = accumulated + suffix;
      } else if (!streamDone && accumulated) {
        if (botIdx !== -1) last[botIdx].content = accumulated;
      } else if (!streamDone && botIdx !== -1 && last[botIdx].content === "…") {
        setBotContent("Sem resposta.");
        last[botIdx].content = "Sem resposta.";
      }
    } catch (err) {
      setBotContent("Erro ao consultar: " + err.message);
      if (botIdx !== -1) last[botIdx].content = "Erro ao consultar: " + err.message;
    } finally {
      if (submitBtn) submitBtn.disabled = false;
      userInputEl.disabled = false;
    }
  });

  btnTransmutar.addEventListener("click", () => {
    processarForm.reset();
    modalOverlay.hidden = false;
  });

  btnCancelar.addEventListener("click", () => {
    modalOverlay.hidden = true;
  });

  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) modalOverlay.hidden = true;
  });

  processarForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const urlInput = document.getElementById("videoUrl");
    const url = urlInput.value.trim();
    if (!url) return;

    modalOverlay.hidden = true;
    loadingOverlay.hidden = false;

    try {
      const res = await fetch(API_BASE + "/processar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const d = data.detail;
        throw new Error(
          Array.isArray(d) ? (d[0] && d[0].msg) || "Falha ao processar" : typeof d === "string" ? d : "Falha ao processar"
        );
      }
      await fetchVideos();
      if (data.video_id) selectVideo(data.video_id);
    } catch (err) {
      alert("Erro: " + (err.message || "Falha ao processar o vídeo."));
    } finally {
      loadingOverlay.hidden = true;
    }
  });

  fetchVideos();
})();
