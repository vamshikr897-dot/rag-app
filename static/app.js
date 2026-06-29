document.addEventListener("DOMContentLoaded", () => {
  const chatLog = document.getElementById("chat-log");
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question-input");
  const button = document.getElementById("ask-button");
  const gradeSelect = document.getElementById("grade-select");
  const newChatButton = document.getElementById("new-chat-button");
  const micButton = document.getElementById("mic-button");
  const voiceStatus = document.getElementById("voice-status");
  let currentGrade = null;
  let activeAbortController = null;
  let conversationHistory = [];
  const MAX_HISTORY_MESSAGES = 6;
  const HISTORY_CONTENT_CAP = 4000;
  const MAX_REGENERATE_ATTEMPTS = 2;

  let recognition = null;
  let isListening = false;
  let userStoppedListening = true;
  let dictationBaseText = "";
  let finalizedTranscript = "";
  let restartAttempts = 0;
  const MAX_RESTART_ATTEMPTS = 5;
  const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;

  const SESSION_STORAGE_KEY = "curiosityCoachSessionId";
  let sessionId = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }

  const GRADE_STORAGE_KEY = "curiosityCoachGrade";

  function capHistoryContent(text) {
    return text.length > HISTORY_CONTENT_CAP ? `${text.slice(0, HISTORY_CONTENT_CAP)}…` : text;
  }

  const AVATARS = {
    student: "\u{1F9D1}‍\u{1F393}",
    assistant: "\u{1F52C}",
  };

  function createMessageRow(roleClass, bubbleEl) {
    const row = document.createElement("div");
    row.className = `message-row ${roleClass}`;

    const avatar = document.createElement("div");
    const avatarRole = roleClass === "thinking" ? "assistant" : roleClass;
    avatar.className = `avatar avatar-${avatarRole}`;
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = AVATARS[avatarRole];

    row.appendChild(avatar);
    row.appendChild(bubbleEl);
    return row;
  }

  function appendMessage(text, className) {
    const bubble = document.createElement("div");
    bubble.className = `message ${className}`;
    bubble.textContent = text;

    const node = className === "student" ? createMessageRow("student", bubble) : bubble;

    chatLog.appendChild(node);
    chatLog.scrollTop = chatLog.scrollHeight;
    return bubble;
  }

  function appendThinkingIndicator() {
    const bubble = document.createElement("div");
    bubble.className = "message thinking";
    bubble.setAttribute("role", "status");
    bubble.setAttribute("aria-label", "Assistant is thinking");

    const dots = document.createElement("div");
    dots.className = "thinking-dots";
    for (let i = 0; i < 3; i++) {
      dots.appendChild(document.createElement("span"));
    }
    bubble.appendChild(dots);

    const row = createMessageRow("thinking", bubble);
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
    return row;
  }

  // Renders **bold**/__bold__ and *italic*/_italic_ spans as real elements;
  // everything else becomes a plain text node, so untrusted text is never
  // parsed as HTML (no innerHTML use, matching the rest of this file).
  // LaTeX math spans (\[...\], \(...\), $$...$$) are matched first and passed through
  // verbatim -- otherwise the italic regex's bare "_" would wrongly match LaTeX
  // subscripts like K_{\text{final}} before KaTeX's auto-render ever sees them.
  function renderInlineMarkdown(text, parentEl) {
    const pattern = /(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$)|(\*\*|__)(.+?)\2|(\*|_)(.+?)\4/g;
    let lastIndex = 0;
    let match;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parentEl.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      if (match[1]) {
        parentEl.appendChild(document.createTextNode(match[1]));
      } else if (match[2]) {
        const strong = document.createElement("strong");
        strong.textContent = match[3];
        parentEl.appendChild(strong);
      } else {
        const em = document.createElement("em");
        em.textContent = match[5];
        parentEl.appendChild(em);
      }
      lastIndex = pattern.lastIndex;
    }
    if (lastIndex < text.length) {
      parentEl.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function renderMarkdownToFragment(markdownText) {
    const fragment = document.createDocumentFragment();
    const lines = markdownText.split("\n");
    const bulletRe = /^\s*[-*]\s+(.*)$/;
    const numberedRe = /^\s*\d+\.\s+(.*)$/;

    let i = 0;
    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) {
        i++;
        continue;
      }

      if (bulletRe.test(line) || numberedRe.test(line)) {
        const listRe = bulletRe.test(line) ? bulletRe : numberedRe;
        const list = document.createElement(listRe === bulletRe ? "ul" : "ol");
        let currentLi = null;

        while (i < lines.length) {
          if (!lines[i].trim()) {
            // Blank line: only treat it as the end of the list if the list doesn't
            // resume after it (LLM output often puts a blank line between items).
            let j = i + 1;
            while (j < lines.length && !lines[j].trim()) {
              j++;
            }
            if (j < lines.length && listRe.test(lines[j])) {
              i = j;
              continue;
            }
            break;
          }

          const itemMatch = lines[i].match(listRe);
          if (itemMatch) {
            currentLi = document.createElement("li");
            renderInlineMarkdown(itemMatch[1], currentLi);
            list.appendChild(currentLi);
            i++;
          } else if (currentLi && /^\s/.test(lines[i]) && !bulletRe.test(lines[i]) && !numberedRe.test(lines[i])) {
            // Indented continuation line under the current item (e.g. an "Examples:" sub-line) -
            // keep it in the same <li> instead of letting it break the list into two separate lists.
            currentLi.appendChild(document.createElement("br"));
            renderInlineMarkdown(lines[i].trim(), currentLi);
            i++;
          } else {
            break;
          }
        }
        fragment.appendChild(list);
        continue;
      }

      const paragraphLines = [];
      while (i < lines.length && lines[i].trim() && !bulletRe.test(lines[i]) && !numberedRe.test(lines[i])) {
        paragraphLines.push(lines[i]);
        i++;
      }
      const p = document.createElement("p");
      renderInlineMarkdown(paragraphLines.join(" "), p);
      fragment.appendChild(p);
    }

    return fragment;
  }

  function dedupeSources(sources) {
    const seen = new Map();
    for (const source of sources) {
      const key = `${source.chapter_number ?? source.chapter_title}|${source.page}`;
      if (!seen.has(key)) {
        seen.set(key, source);
      }
    }
    return Array.from(seen.values()).sort((a, b) => {
      const an = a.chapter_number ?? Infinity;
      const bn = b.chapter_number ?? Infinity;
      if (an !== bn) {
        return an - bn;
      }
      return a.page - b.page;
    });
  }

  function formatPageRanges(pages) {
    const sorted = [...new Set(pages)].sort((a, b) => a - b);
    const ranges = [];
    let start = sorted[0];
    let prev = sorted[0];

    for (let i = 1; i <= sorted.length; i++) {
      const current = sorted[i];
      if (current === prev + 1) {
        prev = current;
        continue;
      }
      ranges.push(start === prev ? `${start}` : `${start}–${prev}`);
      start = current;
      prev = current;
    }

    const label = sorted.length === 1 ? "p." : "pp.";
    return `${label}${ranges.join(", ")}`;
  }

  function groupSourcesByChapter(sources) {
    const groups = new Map();
    for (const source of sources) {
      const key = source.chapter_number ?? source.chapter_title;
      if (!groups.has(key)) {
        groups.set(key, {
          chapter_number: source.chapter_number,
          chapter_title: source.chapter_title,
          pages: [],
        });
      }
      groups.get(key).pages.push(source.page);
    }

    return Array.from(groups.values()).sort((a, b) => {
      const an = a.chapter_number ?? Infinity;
      const bn = b.chapter_number ?? Infinity;
      return an - bn;
    });
  }

  function appendAssistantAnswer(result, question, attemptCount = 0) {
    const bubble = document.createElement("div");
    bubble.className = "message assistant";

    if (result.chapter_heading) {
      const heading = document.createElement("div");
      heading.className = "chapter-heading";
      heading.textContent = result.chapter_heading;
      bubble.appendChild(heading);
    }

    const answer = document.createElement("div");
    answer.className = "answer-body";
    answer.appendChild(renderMarkdownToFragment(result.answer));
    bubble.appendChild(answer);

    if (window.renderMathInElement) {
      renderMathInElement(answer, {
        delimiters: [
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$$", right: "$$", display: true },
        ],
        throwOnError: false,
      });
    }

    const dedupedSources = result.sources ? dedupeSources(result.sources) : [];
    const groupedSources = groupSourcesByChapter(dedupedSources);
    if (groupedSources.length > 0) {
      const sources = document.createElement("div");
      sources.className = "sources";

      const label = document.createElement("div");
      label.textContent = "Sources:";
      sources.appendChild(label);

      const list = document.createElement("ul");
      for (const group of groupedSources) {
        const item = document.createElement("li");
        const chapter = group.chapter_number
          ? `Chapter ${group.chapter_number}`
          : group.chapter_title;
        item.textContent = `${chapter} (${group.chapter_title}), ${formatPageRanges(group.pages)}`;
        list.appendChild(item);
      }
      sources.appendChild(list);
      bubble.appendChild(sources);
    }

    if (result.suggestions && result.suggestions.length > 0) {
      const chips = document.createElement("div");
      chips.className = "example-chips";
      for (const suggestion of result.suggestions) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "example-chip";
        chip.textContent = suggestion;
        chip.addEventListener("click", () => {
          input.value = suggestion;
          if (!input.disabled) {
            form.requestSubmit();
          }
        });
        chips.appendChild(chip);
      }
      bubble.appendChild(chips);
    }

    if (result.interaction_id) {
      appendFeedbackControls(bubble, result, question, attemptCount);
    }

    const row = createMessageRow("assistant", bubble);
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  const FEEDBACK_REASONS = [
    { value: "not_relevant", label: "Didn't answer my question" },
    { value: "too_complicated", label: "Too hard to understand" },
    { value: "too_short", label: "Too short" },
    { value: "other", label: "Something else" },
  ];

  function sendFeedback(interactionId, vote, reason, detail) {
    fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interaction_id: interactionId, vote, reason: reason || null, detail: detail || null }),
    }).catch(() => {});
  }

  function appendFeedbackControls(bubble, result, question, attemptCount) {
    const feedbackEl = document.createElement("div");
    feedbackEl.className = "feedback-actions";

    const topRow = document.createElement("div");
    topRow.className = "feedback-top-row";

    const extraEl = document.createElement("div");
    extraEl.className = "feedback-extra";

    function appendNote(text) {
      const note = document.createElement("div");
      note.className = "feedback-note";
      note.textContent = text;
      extraEl.appendChild(note);
    }

    function triggerRegeneration(reason, detail) {
      sendFeedback(result.interaction_id, "down", reason, detail);

      if (attemptCount >= MAX_REGENERATE_ATTEMPTS) {
        appendNote("Thanks for the feedback!");
        return;
      }
      appendNote("Thanks — let me try again...");

      submitQuestion(question, {
        isRegeneration: true,
        attemptCount: attemptCount + 1,
        regenerate: {
          reason,
          detail: detail || null,
          previous_answer: result.answer,
          previous_interaction_id: result.interaction_id,
        },
      });
    }

    function showReasonChips() {
      const prompt = document.createElement("div");
      prompt.className = "feedback-prompt";
      prompt.textContent = "Sorry about that — what wasn't quite right?";
      extraEl.appendChild(prompt);

      const reasonsEl = document.createElement("div");
      reasonsEl.className = "feedback-reasons";

      const chips = [];
      for (const reasonOption of FEEDBACK_REASONS) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "example-chip";
        chip.textContent = reasonOption.label;
        chip.addEventListener("click", () => {
          chips.forEach((c) => {
            c.disabled = true;
          });
          chip.classList.add("selected");

          if (reasonOption.value === "other") {
            showDetailInput();
          } else {
            triggerRegeneration(reasonOption.value, null);
          }
        });
        chips.push(chip);
        reasonsEl.appendChild(chip);
      }
      extraEl.appendChild(reasonsEl);
    }

    function showDetailInput() {
      const detailForm = document.createElement("form");
      detailForm.className = "feedback-detail-form";

      const detailInput = document.createElement("input");
      detailInput.type = "text";
      detailInput.maxLength = 300;
      detailInput.placeholder = "Tell me a bit more...";

      const detailSubmit = document.createElement("button");
      detailSubmit.type = "submit";
      detailSubmit.setAttribute("aria-label", "Send feedback detail");
      detailSubmit.textContent = "Send";

      detailForm.appendChild(detailInput);
      detailForm.appendChild(detailSubmit);
      detailForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const detail = detailInput.value.trim();
        detailInput.disabled = true;
        detailSubmit.disabled = true;
        triggerRegeneration("other", detail);
      });

      extraEl.appendChild(detailForm);
      detailInput.focus();
    }

    const label = document.createElement("span");
    label.className = "feedback-label";
    label.textContent = "Was this answer helpful?";

    const upBtn = document.createElement("button");
    upBtn.type = "button";
    upBtn.className = "feedback-btn feedback-btn-up";
    upBtn.title = "This helped me!";
    upBtn.setAttribute("aria-label", "This helped me!");
    upBtn.textContent = "👍";
    upBtn.addEventListener("click", () => {
      upBtn.disabled = true;
      downBtn.disabled = true;
      upBtn.classList.add("selected");
      sendFeedback(result.interaction_id, "up");
      appendNote("Glad that helped! Ask me anything else. 🎉");
    });

    const downBtn = document.createElement("button");
    downBtn.type = "button";
    downBtn.className = "feedback-btn feedback-btn-down";
    downBtn.title = "This didn't help";
    downBtn.setAttribute("aria-label", "This didn't help");
    downBtn.textContent = "👎";
    downBtn.addEventListener("click", () => {
      upBtn.disabled = true;
      downBtn.disabled = true;
      downBtn.classList.add("selected");
      showReasonChips();
    });

    topRow.appendChild(label);
    topRow.appendChild(upBtn);
    topRow.appendChild(downBtn);

    feedbackEl.appendChild(topRow);
    feedbackEl.appendChild(extraEl);
    bubble.appendChild(feedbackEl);
  }

  const GRADE_ICONS = { 6: "\u{1F52D}", 7: "\u{1F331}", 8: "\u{1F9EA}", 9: "⚛️", 10: "\u{1F680}" };
  const GRADE_THEME_CLASSES = [6, 7, 8, 9, 10].map((g) => `grade-theme-${g}`);

  async function loadGrades() {
    button.disabled = true;
    input.disabled = true;
    micButton.disabled = true;
    gradeSelect.disabled = true;

    try {
      const response = await fetch("/api/grades");
      if (!response.ok) {
        throw new Error("bad status");
      }
      const body = await response.json();

      for (const grade of body.grades) {
        const option = document.createElement("option");
        option.value = String(grade);
        const icon = GRADE_ICONS[grade] ? `${GRADE_ICONS[grade]} ` : "";
        option.textContent = `${icon}Grade ${grade}`;
        gradeSelect.appendChild(option);
      }
      gradeSelect.disabled = false;

      const savedGrade = Number(localStorage.getItem(GRADE_STORAGE_KEY));
      if (savedGrade && body.grades.includes(savedGrade)) {
        gradeSelect.value = String(savedGrade);
        currentGrade = savedGrade;
        applyGradeTheme(currentGrade);
        input.disabled = false;
        button.disabled = false;
        micButton.disabled = false;
      }
      renderWelcomeScreen();
    } catch {
      appendMessage("Couldn't load grade list. Please refresh the page.", "error");
    }
  }

  function applyGradeTheme(grade) {
    document.body.classList.remove(...GRADE_THEME_CLASSES);
    if (grade !== null) {
      document.body.classList.add(`grade-theme-${grade}`);
    }
  }

  const EXAMPLE_QUESTIONS_BY_GRADE = {
    6: [
      "How do magnets attract things?",
      "Why does water turn to ice?",
      "How do we separate sand and water?",
    ],
    7: [
      "Why does lemon juice redden litmus?",
      "Why does puberty change us?",
      "Why do eclipses happen?",
    ],
    8: [
      "Why is wind strong some days?",
      "What is a saturated solution?",
      "How do electromagnets work?",
    ],
    9: [
      "What is photosynthesis?",
      "Why do we hear an echo indoors?",
      "What lets a cell make energy?",
    ],
    10: [
      "Why does a knife look bent in water?",
      "Why is the sky blue?",
      "Why does rusting happen faster at sea?",
    ],
  };

  function renderWelcomeScreen() {
    chatLog.textContent = "";

    const welcome = document.createElement("div");
    welcome.className = "welcome-screen";

    const emoji = document.createElement("div");
    emoji.className = "welcome-emoji";
    emoji.textContent = AVATARS.assistant;
    emoji.setAttribute("aria-hidden", "true");
    welcome.appendChild(emoji);

    const heading = document.createElement("h2");
    heading.textContent = "Hi, I'm Curiosity Coach!";
    welcome.appendChild(heading);

    if (currentGrade === null) {
      const prompt = document.createElement("p");
      prompt.className = "grade-prompt";
      prompt.textContent = "👆 Select your grade above to get started!";
      welcome.appendChild(prompt);
    } else {
      const sub = document.createElement("p");
      sub.textContent = "There's no such thing as a silly question — ask me anything, or try one of these:";
      welcome.appendChild(sub);

      const chips = document.createElement("div");
      chips.className = "example-chips";
      const exampleQuestions = EXAMPLE_QUESTIONS_BY_GRADE[currentGrade] || [];
      for (const question of exampleQuestions) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "example-chip";
        chip.textContent = question;
        chip.addEventListener("click", () => {
          input.value = question;
          if (!input.disabled) {
            form.requestSubmit();
          }
        });
        chips.appendChild(chip);
      }
      welcome.appendChild(chips);
    }

    chatLog.appendChild(welcome);
  }

  function clearWelcomeScreenIfPresent() {
    const welcome = chatLog.querySelector(".welcome-screen");
    if (welcome) {
      welcome.remove();
    }
  }

  gradeSelect.addEventListener("change", () => {
    if (isListening) {
      stopListening();
    }

    const newGrade = gradeSelect.value ? Number(gradeSelect.value) : null;

    if (currentGrade !== null && newGrade !== currentGrade) {
      appendMessage(`Switched to Grade ${newGrade}.`, "info");
    }

    currentGrade = newGrade;
    if (newGrade !== null) {
      localStorage.setItem(GRADE_STORAGE_KEY, String(newGrade));
    } else {
      localStorage.removeItem(GRADE_STORAGE_KEY);
    }
    conversationHistory = [];
    applyGradeTheme(currentGrade);
    const hasGrade = currentGrade !== null;
    input.disabled = !hasGrade;
    button.disabled = !hasGrade;
    micButton.disabled = !hasGrade;
    if (chatLog.querySelector(".welcome-screen")) {
      renderWelcomeScreen();
    }
    if (hasGrade) {
      input.focus();
    }
  });

  newChatButton.addEventListener("click", () => {
    if (isListening) {
      stopListening();
    }
    if (activeAbortController) {
      activeAbortController.abort();
    }
    conversationHistory = [];
    renderWelcomeScreen();
    input.value = "";
    const hasGrade = currentGrade !== null;
    input.disabled = !hasGrade;
    button.disabled = !hasGrade;
    micButton.disabled = !hasGrade;
    if (hasGrade) {
      input.focus();
    }
  });

  function setVoiceStatus(message, isError) {
    voiceStatus.textContent = message;
    voiceStatus.classList.toggle("voice-status-error", Boolean(isError));
    voiceStatus.classList.toggle("sr-only", !message);
  }

  function setListeningUIState(listening) {
    isListening = listening;
    micButton.classList.toggle("listening", listening);
    micButton.setAttribute("aria-label", listening ? "Stop voice input" : "Use voice input");
    micButton.setAttribute("title", listening ? "Listening… click to stop" : "Click and start speaking");
    if (!listening) {
      setVoiceStatus("", false);
    }
  }

  function joinDictation(base, finalText, interimText) {
    const dictated = `${finalText}${interimText}`;
    if (!base) {
      return dictated;
    }
    if (!dictated) {
      return base;
    }
    const needsSpace = !/\s$/.test(base);
    return `${base}${needsSpace ? " " : ""}${dictated}`;
  }

  function handleRecognitionResult(event) {
    restartAttempts = 0;

    let finalChunk = "";
    let interimChunk = "";

    for (let i = 0; i < event.results.length; i++) {
      const result = event.results[i];
      const transcript = result[0].transcript;
      if (result.isFinal) {
        finalChunk += transcript;
      } else {
        interimChunk += transcript;
      }
    }

    finalizedTranscript = finalChunk;
    input.value = joinDictation(dictationBaseText, finalizedTranscript, interimChunk);
  }

  function handleRecognitionError(event) {
    console.error("Voice input error:", event.error);

    if (event.error === "no-speech" || event.error === "aborted") {
      return;
    }

    if (
      event.error === "not-allowed" ||
      event.error === "permission-denied" ||
      event.error === "service-not-allowed"
    ) {
      userStoppedListening = true;
      setListeningUIState(false);
      setVoiceStatus(
        "Microphone access was denied. Check your browser permissions to use voice input.",
        true
      );
      return;
    }

    userStoppedListening = true;
    setListeningUIState(false);
    setVoiceStatus("Voice input had a problem and stopped. You can try again or type your question.", true);
  }

  function handleRecognitionEnd() {
    if (userStoppedListening) {
      setListeningUIState(false);
      recognition = null;
      return;
    }

    if (restartAttempts >= MAX_RESTART_ATTEMPTS) {
      setListeningUIState(false);
      setVoiceStatus("Voice input stopped. Tap the mic to try again.", true);
      recognition = null;
      return;
    }

    restartAttempts++;
    try {
      recognition = createRecognition();
      recognition.start();
    } catch (err) {
      console.error("Voice input failed to restart:", err);
      setListeningUIState(false);
      setVoiceStatus("Voice input stopped unexpectedly. Tap the mic to try again.", true);
      recognition = null;
    }
  }

  function createRecognition() {
    const recog = new SpeechRecognitionImpl();
    recog.continuous = true;
    recog.interimResults = true;
    recog.lang = "en-US";

    recog.onresult = handleRecognitionResult;
    recog.onerror = handleRecognitionError;
    recog.onend = handleRecognitionEnd;

    return recog;
  }

  function startListening() {
    dictationBaseText = input.value;
    finalizedTranscript = "";
    userStoppedListening = false;
    restartAttempts = 0;

    try {
      recognition = createRecognition();
      recognition.start();
    } catch (err) {
      console.error("Voice input failed to start:", err);
      recognition = null;
      setListeningUIState(false);
      setVoiceStatus("Couldn't start voice input. Please try again.", true);
      return;
    }

    setListeningUIState(true);
    setVoiceStatus("Listening...", false);
  }

  function stopListening() {
    userStoppedListening = true;
    if (recognition) {
      recognition.stop();
    }
    setListeningUIState(false);
  }

  function handleMicButtonClick() {
    if (micButton.disabled) {
      return;
    }
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }

  function initVoiceInput() {
    if (!SpeechRecognitionImpl) {
      console.info("Voice input: SpeechRecognition API not detected in this browser.");
      micButton.hidden = true;
      return;
    }
    console.info("Voice input: SpeechRecognition API detected, mic button enabled.");
    micButton.hidden = false;
    micButton.addEventListener("click", handleMicButtonClick);
  }

  loadGrades();
  initVoiceInput();

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!input.disabled) {
        form.requestSubmit();
      }
    }
  });

  async function submitQuestion(question, { regenerate = null, isRegeneration = false, attemptCount = 0 } = {}) {
    if (!isRegeneration) {
      clearWelcomeScreenIfPresent();
      appendMessage(question, "student");
    }
    button.disabled = true;
    input.disabled = true;
    micButton.disabled = true;

    const thinkingEl = appendThinkingIndicator();
    activeAbortController = new AbortController();

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          grade: currentGrade,
          session_id: sessionId,
          history: conversationHistory,
          ...(regenerate ? { regenerate } : {}),
        }),
        signal: activeAbortController.signal,
      });

      thinkingEl.remove();

      let body;
      try {
        body = await response.json();
      } catch {
        body = null;
      }

      if (!response.ok) {
        let message = body && body.error;
        if (!message && body && Array.isArray(body.detail) && body.detail[0] && body.detail[0].msg) {
          message = body.detail[0].msg;
        }
        appendMessage(message || "Something went wrong. Please try again.", "error");
      } else {
        appendAssistantAnswer(body, question, attemptCount);
        if (body.answer && body.answer.trim()) {
          conversationHistory.push({ role: "user", content: capHistoryContent(question) });
          conversationHistory.push({ role: "assistant", content: capHistoryContent(body.answer) });
          if (conversationHistory.length > MAX_HISTORY_MESSAGES) {
            conversationHistory = conversationHistory.slice(-MAX_HISTORY_MESSAGES);
          }
        }
      }
    } catch (err) {
      thinkingEl.remove();
      if (err.name !== "AbortError") {
        appendMessage("Couldn't reach the server. Check your connection and try again.", "error");
      }
    } finally {
      activeAbortController = null;
      const hasGrade = currentGrade !== null;
      button.disabled = !hasGrade;
      input.disabled = !hasGrade;
      micButton.disabled = !hasGrade;
      if (hasGrade) {
        input.focus();
      }
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();

    if (isListening) {
      stopListening();
    }

    const question = input.value.trim();
    if (!question) {
      return;
    }
    if (currentGrade === null) {
      appendMessage("Please choose your grade first.", "error");
      return;
    }

    input.value = "";
    submitQuestion(question);
  });
});
