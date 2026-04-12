"use strict";

const STORAGE = {
  chat: "constitutional_ai.chat.v2",
  transcript: "constitutional_ai.transcript.v2",
};

const PROVIDERS = [
  { id: "openai", label: "OpenAI", credential: "openai_api_key", requiresKey: true, defaultModel: "gpt-4o-mini", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM uses the standard OpenAI endpoint automatically." },
  { id: "anthropic", label: "Anthropic", credential: "anthropic_api_key", requiresKey: true, defaultModel: "claude-sonnet-4-5-20250929", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM uses Anthropic's standard endpoint automatically." },
  { id: "gemini", label: "Gemini", credential: "gemini_api_key", requiresKey: true, defaultModel: "gemini-2.5-flash", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM routes Gemini requests directly from the provider and API key." },
  { id: "xai", label: "xAI", credential: "xai_api_key", requiresKey: true, defaultModel: "grok-2-latest", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM uses xAI's standard endpoint automatically." },
  { id: "openrouter", label: "OpenRouter", credential: "openrouter_api_key", requiresKey: true, defaultModel: "openai/gpt-4o-mini", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM uses OpenRouter's standard endpoint automatically." },
  { id: "groq", label: "Groq", credential: "groq_api_key", requiresKey: true, defaultModel: "llama-3.3-70b-versatile", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM uses Groq's standard endpoint automatically." },
  { id: "togetherai", label: "Together AI", credential: "togetherai_api_key", requiresKey: true, defaultModel: "meta-llama/Llama-3.3-70B-Instruct-Turbo", defaultApiBase: "", defaultApiVersion: "", showApiBase: false, showApiVersion: false, apiBaseHelp: "LiteLLM uses Together AI's standard endpoint automatically." },
  { id: "huggingface", label: "Hugging Face", credential: "huggingface_api_key", requiresKey: true, defaultModel: "meta-llama/Meta-Llama-3.1-8B-Instruct", defaultApiBase: "", defaultApiVersion: "", showApiBase: true, showApiVersion: false, apiBaseHelp: "Set API base only when using a dedicated Hugging Face Inference Endpoint." },
  { id: "azure", label: "Azure OpenAI", credential: "azure_api_key", requiresKey: true, defaultModel: "gpt-4o-mini", defaultApiBase: "", defaultApiVersion: "", showApiBase: true, showApiVersion: true, apiBaseHelp: "Azure requires your resource endpoint and usually an API version." },
  { id: "ollama", label: "Ollama", credential: null, requiresKey: false, defaultModel: "llama3.2", defaultApiBase: "http://localhost:11434", defaultApiVersion: "", showApiBase: true, showApiVersion: false, apiBaseHelp: "Local Ollama needs the local server URL." },
  { id: "lm_studio", label: "LM Studio", credential: null, requiresKey: false, defaultModel: "local-model", defaultApiBase: "http://localhost:1234", defaultApiVersion: "", showApiBase: true, showApiVersion: false, apiBaseHelp: "Local LM Studio needs the local server URL." },
];

const CREDENTIAL_INPUTS = {
  openai_api_key: "cred-openai",
  anthropic_api_key: "cred-anthropic",
  gemini_api_key: "cred-gemini",
  xai_api_key: "cred-xai",
  openrouter_api_key: "cred-openrouter",
  groq_api_key: "cred-groq",
  togetherai_api_key: "cred-togetherai",
  huggingface_api_key: "cred-huggingface",
  azure_api_key: "cred-azure",
};

const state = {
  config: null,
  meta: null,
  models: {
    writer: { models: [], supportsListing: false, manual: false },
    judge: { models: [], supportsListing: false, manual: false },
  },
  chat: loadJson(STORAGE.chat, []),
  transcript: loadJson(STORAGE.transcript, []),
  busy: false,
  currentTurnId: null,
};

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : structuredClone(fallback);
  } catch {
    return structuredClone(fallback);
  }
}

function saveJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function providerMeta(providerId) {
  return PROVIDERS.find((provider) => provider.id === providerId) || PROVIDERS[0];
}

function defaultConfig() {
  return {
    settings: {
      credentials: {
        openai_api_key: "",
        anthropic_api_key: "",
        gemini_api_key: "",
        xai_api_key: "",
        openrouter_api_key: "",
        groq_api_key: "",
        togetherai_api_key: "",
        huggingface_api_key: "",
        azure_api_key: "",
      },
      writer: {
        provider: "openai",
        model: "gpt-4o-mini",
        api_base: "",
        api_version: "",
      },
      judge: {
        provider: "openai",
        model: "gpt-4o-mini",
        api_base: "",
        api_version: "",
      },
      temperature: 0.4,
      max_tokens: 650,
      max_revisions_per_rule: 1,
      execution_mode: "sequential",
      parallel_max_iterations: 0,
      max_iteration_ms: 0,
      timeout_ms: 45000,
    },
    rules: [
      "Be helpful, clear, and accurate.",
      "Do not provide illegal wrongdoing instructions or facilitation.",
      "Do not reveal secrets or private data (including API keys).",
      "If uncertain, state uncertainty and ask clarifying questions.",
      "Keep responses concise unless the user asks for depth.",
    ],
    prompts: {
      writer_system:
        "You are the writer agent. Revise the existing response with minimal changes. Preserve the original wording, structure, and tone as much as possible. Only modify the specific parts needed to address the judge's critique and follow the provided rule. Do not rewrite or rephrase unaffected sections. Return ONLY the final user-facing answer, with no meta-commentary.",
      judge_pass_system:
        'You are the judge agent. Evaluate the writer agent\'s answer against the given rule ONLY. Do not use any other criteria. Return JSON ONLY (no markdown, no extra text). First decide whether the rule applies to this user prompt and answer. If it does not apply, mark it as not applicable. If it applies, decide whether the answer follows the rule. Schema: {"applies": boolean, "pass": boolean}. Constraints: if applies is false, pass MUST be true.',
      judge_critique_system:
        'You are the judge agent. Evaluate the writer agent\'s answer against the given rule ONLY. The answer has already failed this rule. Provide a concise critique and explicit, actionable required fixes. Base your judgment only on the given rule, not on any other criteria. The required fixes must clearly identify what part of the answer is problematic and how it must be changed so the revised answer no longer violates the rule. Return JSON ONLY (no markdown, no extra text). Schema: {"critique": string, "required_fixes": string}.',
    },
  };
}

function truncate(text, max) {
  const value = String(text || "");
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function escapeHtml(text) {
  const node = document.createElement("div");
  node.textContent = String(text ?? "");
  return node.innerHTML;
}

function setPage(page) {
  for (const section of document.querySelectorAll(".page")) {
    section.hidden = section.id !== `page-${page}`;
  }
  for (const tab of document.querySelectorAll(".tab")) {
    tab.setAttribute("aria-selected", tab.dataset.page === page ? "true" : "false");
  }
}

function setSettingsSection(sectionName) {
  for (const button of document.querySelectorAll(".settingsNavButton")) {
    button.setAttribute("aria-selected", button.dataset.settingsSection === sectionName ? "true" : "false");
  }
  for (const panel of document.querySelectorAll(".settingsPanel")) {
    panel.hidden = panel.id !== `settings-panel-${sectionName}`;
  }
}

function setBusy(isBusy, label) {
  state.busy = isBusy;
  const pill = document.getElementById("status-pill");
  pill.textContent = label || (isBusy ? "working..." : "idle");
  pill.classList.toggle("ok", !isBusy);
  pill.classList.toggle("bad", isBusy);

  for (const id of [
    "btn-send",
    "btn-save-config",
    "btn-save-rules",
    "btn-save-settings",
    "btn-writer-refresh-models",
    "btn-judge-refresh-models",
    "btn-writer-test-connection",
    "btn-judge-test-connection",
  ]) {
    const el = document.getElementById(id);
    if (el) el.disabled = isBusy;
  }
  const stop = document.getElementById("btn-stop");
  if (stop) stop.disabled = !isBusy || !state.currentTurnId;
}

function statusLabelFromEvent(event) {
  if (!event || !event.stage) return "running constitutional loop...";
  const stageMap = {
    initial_started: "Preparing initial draft",
    initial_completed: "Initial draft ready",
    parallel_started: "Parallel mode started",
    parallel_pass_checks_started: "Parallel: checking all rules",
    parallel_pass_checks_completed: "Parallel: rule checks complete",
    parallel_critique_started: "Parallel: critiquing failed rules",
    parallel_critique_completed: "Parallel: critiques complete",
    parallel_revision_started: "Parallel: revising answer",
    parallel_revision_completed: "Parallel: revision complete",
    parallel_iteration_limit_reached: "Parallel: iteration limit reached",
    parallel_completed: "Parallel: all rules satisfied",
    sequential_started: "Sequential mode started",
    sequential_check_started: "Sequential: checking rule",
    sequential_not_applicable: "Sequential: rule not applicable",
    sequential_passed: "Sequential: rule passed",
    sequential_failed: "Sequential: rule failed",
    sequential_revision_started: "Sequential: revising for rule",
    sequential_revision_completed: "Sequential: revision complete for rule",
    sequential_revision_limit_reached: "Sequential: revision limit reached for rule",
    sequential_completed: "Sequential: all checks complete",
    turn_stopped: "Stopped by user",
    turn_timed_out: "Stopped by time limit",
    turn_completed: "Completed",
  };
  const base = stageMap[event.stage] || "Running";
  const rulePart = Number.isFinite(event.rule_index) ? ` (rule ${event.rule_index + 1})` : "";
  const iterPart = Number.isFinite(event.iteration) ? ` (round ${event.iteration + 1})` : "";
  return `${base}${rulePart}${iterPart}`;
}

function normalizeChat(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .map((msg) => {
      if (!msg || (msg.role !== "user" && msg.role !== "assistant")) return null;
      return {
        role: msg.role,
        content: String(msg.content || ""),
        at: typeof msg.at === "string" ? msg.at : new Date().toISOString(),
      };
    })
    .filter(Boolean);
}

function renderChat() {
  const root = document.getElementById("chat-list");
  root.innerHTML = "";
  for (const msg of state.chat) {
    const div = document.createElement("div");
    div.className = `msg ${msg.role}`;
    div.innerHTML = `
      <div class="meta">
        <span>${escapeHtml(msg.role)}</span>
        <span>${escapeHtml(msg.at)}</span>
      </div>
      <div class="content"></div>
    `;
    div.querySelector(".content").textContent = msg.content;
    root.appendChild(div);
  }
}

function renderTranscript() {
  const root = document.getElementById("transcript-list");
  root.innerHTML = "";
  if (!state.transcript.length) {
    const empty = document.createElement("div");
    empty.className = "small";
    empty.textContent = "No transcript yet.";
    root.appendChild(empty);
    return;
  }

  for (const turn of state.transcript) {
    const details = document.createElement("details");
    const failed = (turn.judge?.checks || []).filter((c) => c.applies !== false && c.pass === false).length;
    const usage = turn.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    const duration = Number(turn.duration_ms || 0);
    const summary = document.createElement("summary");
    summary.textContent = `User: ${truncate(turn.user || "", 70)} | checks: ${(turn.judge?.checks || []).length} | failed: ${failed} | total tokens: ${usage.total_tokens} | duration: ${duration} ms`;
    details.appendChild(summary);

    const drafts = turn.writer?.drafts || [];
    const initialDraft = drafts.find((draft) => draft.kind === "initial");
    const revisionDrafts = drafts.filter((draft) => draft.kind === "revision");

    const renderDraftBlock = (label, content) => {
      const block = document.createElement("div");
      block.className = "msg";
      block.style.marginTop = "8px";
      block.innerHTML = `<div class="small"></div><div class="content"></div>`;
      block.querySelector(".small").textContent = label;
      block.querySelector(".content").textContent = content || "";
      return block;
    };

    details.appendChild(renderDraftBlock("Final answer", turn.final || ""));

    if (initialDraft?.content) {
      const initial = document.createElement("details");
      initial.style.marginTop = "8px";
      initial.innerHTML = `<summary>Initial draft</summary>`;
      initial.appendChild(renderDraftBlock("Initial writer response", initialDraft.content));
      details.appendChild(initial);
    }

    const events = turn.run?.events || [];
    if (events.length) {
      const timeline = document.createElement("details");
      timeline.style.marginTop = "8px";
      timeline.innerHTML = `<summary>Run stages (${events.length})</summary><div class="small" style="margin-top:6px"></div>`;
      const timelineBody = timeline.querySelector(".small");
      for (const event of events) {
        const item = document.createElement("div");
        const rulePart = Number.isFinite(event.rule_index) ? ` | rule ${event.rule_index + 1}` : "";
        const iterPart = Number.isFinite(event.iteration) ? ` | iteration ${event.iteration + 1}` : "";
        item.textContent = `${event.stage}${rulePart}${iterPart}: ${event.message}`;
        timelineBody.appendChild(item);
      }
      details.appendChild(timeline);
    }

    const renderRuleCheck = (check) => {
      const checkNode = document.createElement("details");
      checkNode.style.marginTop = "8px";
      const status = check.applies === false ? "N/A" : check.pass ? "PASS" : "FAIL";
      const summaryNode = document.createElement("summary");
      const ruleNum = Number.isFinite(check.rule_index) ? check.rule_index + 1 : "?";
      summaryNode.textContent = `Rule ${ruleNum}: ${status}`;
      checkNode.appendChild(summaryNode);

      const content = document.createElement("div");
      content.className = "small";
      content.style.marginTop = "6px";
      content.textContent = check.rule || "";
      checkNode.appendChild(content);

      if (check.applies !== false && check.pass === false) {
        const critique = document.createElement("div");
        critique.className = "msg";
        critique.style.marginTop = "6px";
        critique.innerHTML = `<div class="small">Critique</div><div class="content"></div>`;
        critique.querySelector(".content").textContent = check.critique || "(No critique provided.)";
        checkNode.appendChild(critique);

        const fixes = document.createElement("div");
        fixes.className = "msg";
        fixes.style.marginTop = "6px";
        fixes.innerHTML = `<div class="small">Required fixes</div><div class="content"></div>`;
        fixes.querySelector(".content").textContent = check.required_fixes || "(No required fixes provided.)";
        checkNode.appendChild(fixes);
      }

      const matchingRevisions = revisionDrafts.filter((draft) => {
        if (Number.isFinite(check.iteration) && Number.isFinite(draft.iteration)) {
          return draft.iteration === check.iteration;
        }
        return Number.isFinite(draft.rule_index) && draft.rule_index === check.rule_index;
      });
      for (const [index, draft] of matchingRevisions.entries()) {
        const label =
          matchingRevisions.length > 1
            ? `Revised response ${index + 1}`
            : "Revised response";
        checkNode.appendChild(renderDraftBlock(label, draft.content));
      }
      return checkNode;
    };

    const checks = turn.judge?.checks || [];
    const hasIterationData = checks.some((check) => Number.isFinite(check.iteration));
    if (hasIterationData) {
      const groups = new Map();
      for (const check of checks) {
        const key = Number.isFinite(check.iteration) ? check.iteration : -1;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(check);
      }
      const orderedIterations = Array.from(groups.keys()).sort((a, b) => a - b);
      for (const iteration of orderedIterations) {
        const iterationChecks = groups.get(iteration) || [];
        const failedInIteration = iterationChecks.filter((c) => c.applies !== false && c.pass === false).length;
        const iterationRevisions = revisionDrafts.filter((draft) => Number.isFinite(draft.iteration) && draft.iteration === iteration);
        const iterationNode = document.createElement("details");
        iterationNode.style.marginTop = "8px";
        const label = iteration >= 0 ? `Iteration ${iteration + 1}` : "Checks (unscoped)";
        iterationNode.innerHTML = `<summary>${label} | checks: ${iterationChecks.length} | failed: ${failedInIteration}</summary>`;
        for (const [index, draft] of iterationRevisions.entries()) {
          const revisionLabel =
            iterationRevisions.length > 1
              ? `Revised response ${index + 1} at end of iteration`
              : "Revised response at end of iteration";
          iterationNode.appendChild(renderDraftBlock(revisionLabel, draft.content));
        }
        for (const check of iterationChecks) {
          iterationNode.appendChild(renderRuleCheck(check));
        }
        details.appendChild(iterationNode);
      }
    } else {
      if (revisionDrafts.length) {
        const revisionsNode = document.createElement("details");
        revisionsNode.style.marginTop = "8px";
        revisionsNode.innerHTML = `<summary>Revisions (${revisionDrafts.length})</summary>`;
        for (const [index, draft] of revisionDrafts.entries()) {
          const label =
            Number.isFinite(draft.rule_index)
              ? `Revised response after rule ${draft.rule_index + 1}`
              : `Revised response ${index + 1}`;
          revisionsNode.appendChild(renderDraftBlock(label, draft.content));
        }
        details.appendChild(revisionsNode);
      }
      for (const check of checks) {
        details.appendChild(renderRuleCheck(check));
      }
    }

    root.appendChild(details);
  }
}

function readFormIntoState() {
  if (!state.config) return;

  const credentials = {};
  for (const [fieldName, inputId] of Object.entries(CREDENTIAL_INPUTS)) {
    const value = document.getElementById(inputId).value.trim();
    if (value) credentials[fieldName] = value;
  }

  const readRole = (role) => {
    const modelSelect = document.getElementById(`${role}-model-select`);
    const modelInput = document.getElementById(`${role}-model-input`);
    const manual = !modelInput.classList.contains("hidden");
    return {
      provider: document.getElementById(`${role}-provider`).value,
      model: manual ? modelInput.value.trim() : modelSelect.value.trim(),
      api_base: document.getElementById(`${role}-api-base`).value.trim(),
      api_version: document.getElementById(`${role}-api-version`).value.trim(),
    };
  };

  state.config.settings = {
    credentials,
    writer: readRole("writer"),
    judge: readRole("judge"),
    temperature: Number(document.getElementById("set-temperature").value || 0.4),
    max_tokens: Number(document.getElementById("set-max-tokens").value || 650),
    max_revisions_per_rule: Number(document.getElementById("set-max-revisions").value || 1),
    execution_mode: document.getElementById("set-execution-mode").value || "sequential",
    parallel_max_iterations: Number(document.getElementById("set-parallel-max-iterations").value || 0),
    max_iteration_ms: Number(document.getElementById("set-max-iteration-ms").value || 0),
    timeout_ms: Number(document.getElementById("set-timeout-ms").value || 45000),
  };
  state.config.prompts = {
    writer_system: document.getElementById("prompt-writer").value,
    judge_pass_system: document.getElementById("prompt-pass").value,
    judge_critique_system: document.getElementById("prompt-critique").value,
  };
  state.config.rules = document
    .getElementById("rules-text")
    .value.split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function syncProviderOptions() {
  for (const role of ["writer", "judge"]) {
    const select = document.getElementById(`${role}-provider`);
    select.innerHTML = "";
    for (const provider of PROVIDERS) {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = provider.label;
      select.appendChild(option);
    }
  }
}

function syncModelControl(role) {
  if (!state.config) return;
  const roleConfig = state.config.settings?.[role] || {};
  const info = state.models[role] || { models: [], supportsListing: false, manual: false };
  const modelSelect = document.getElementById(`${role}-model-select`);
  const modelInput = document.getElementById(`${role}-model-input`);
  const status = document.getElementById(`${role}-model-status`);
  const options = Array.from(new Set((info.models || []).map((item) => String(item.id || "").trim()).filter(Boolean)));

  if (info.supportsListing && options.length > 0 && !info.manual) {
    modelSelect.innerHTML = "";
    for (const modelName of options) {
      const option = document.createElement("option");
      option.value = modelName;
      option.textContent = modelName;
      modelSelect.appendChild(option);
    }
    if (roleConfig.model && !options.includes(roleConfig.model)) {
      const option = document.createElement("option");
      option.value = roleConfig.model;
      option.textContent = `${roleConfig.model} (current)`;
      modelSelect.appendChild(option);
    }
    modelSelect.value = roleConfig.model || options[0] || "";
    modelSelect.classList.remove("hidden");
    modelInput.classList.add("hidden");
    modelInput.value = roleConfig.model || "";
    status.textContent = `Model listing is live for ${roleConfig.provider || "this provider"}.`;
  } else {
    modelInput.classList.remove("hidden");
    modelSelect.classList.add("hidden");
    modelInput.value = roleConfig.model || "";
    modelSelect.innerHTML = "";
    status.textContent = info.supportsListing
      ? "No models were returned. Enter a model name manually."
      : "Provider-side model listing is unavailable here. Enter a model name manually.";
  }
}

function syncEndpointFields(role) {
  const provider = providerMeta(document.getElementById(`${role}-provider`).value);
  const apiBaseGroup = document.getElementById(`${role}-api-base-group`);
  const apiVersionGroup = document.getElementById(`${role}-api-version-group`);
  const apiBaseHelp = document.getElementById(`${role}-api-base-help`);
  if (apiBaseGroup) apiBaseGroup.classList.toggle("hidden", !provider.showApiBase);
  if (apiVersionGroup) apiVersionGroup.classList.toggle("hidden", !provider.showApiVersion);
  if (apiBaseHelp) apiBaseHelp.textContent = provider.apiBaseHelp || "";
}

function effectiveCredentialSource(fieldName) {
  return state.meta?.credential_sources?.[fieldName] || "none";
}

function hasCredentialForProvider(providerId) {
  const provider = providerMeta(providerId);
  if (!provider.requiresKey || !provider.credential) return true;
  const typed = document.getElementById(CREDENTIAL_INPUTS[provider.credential])?.value.trim() || "";
  if (typed) return true;
  return effectiveCredentialSource(provider.credential) !== "none";
}

function syncCredentialSourceLabels() {
  for (const [fieldName, inputId] of Object.entries(CREDENTIAL_INPUTS)) {
    const node = document.getElementById(`${inputId}-source`);
    if (!node) continue;
    const typed = document.getElementById(inputId).value.trim();
    if (typed) {
      node.textContent = "This value will be used on save.";
      continue;
    }
    const source = effectiveCredentialSource(fieldName);
    node.textContent =
      source === "environment"
        ? "Source: environment variable"
        : source === "config"
          ? "Source: saved config"
          : "Source: not set";
  }
}

function syncFormFromState() {
  if (!state.config) return;
  const settings = state.config.settings;
  const prompts = state.config.prompts;

  syncProviderOptions();
  document.getElementById("writer-provider").value = settings.writer?.provider || "openai";
  document.getElementById("writer-api-base").value = settings.writer?.api_base || "";
  document.getElementById("writer-api-version").value = settings.writer?.api_version || "";
  document.getElementById("judge-provider").value = settings.judge?.provider || "openai";
  document.getElementById("judge-api-base").value = settings.judge?.api_base || "";
  document.getElementById("judge-api-version").value = settings.judge?.api_version || "";

  document.getElementById("set-temperature").value = String(settings.temperature ?? 0.4);
  document.getElementById("set-max-tokens").value = String(settings.max_tokens ?? 650);
  document.getElementById("set-max-revisions").value = String(settings.max_revisions_per_rule ?? 1);
  document.getElementById("set-execution-mode").value = settings.execution_mode || "sequential";
  document.getElementById("set-parallel-max-iterations").value = String(settings.parallel_max_iterations ?? 0);
  document.getElementById("set-max-iteration-ms").value = String(settings.max_iteration_ms ?? 0);
  document.getElementById("set-timeout-ms").value = String(settings.timeout_ms ?? 45000);
  document.getElementById("rules-text").value = (state.config.rules || []).join("\n");

  document.getElementById("prompt-writer").value = prompts.writer_system || "";
  document.getElementById("prompt-pass").value = prompts.judge_pass_system || "";
  document.getElementById("prompt-critique").value = prompts.judge_critique_system || "";

  syncCredentialSourceLabels();
  syncEndpointFields("writer");
  syncEndpointFields("judge");
  syncModelControl("writer");
  syncModelControl("judge");
  syncStatusPills();
}

function syncStatusPills() {
  if (!state.config) return;
  const writerProvider = document.getElementById("writer-provider").value;
  const judgeProvider = document.getElementById("judge-provider").value;
  const writerModel = currentModelValue("writer");
  const judgeModel = currentModelValue("judge");

  const writerReady = Boolean(writerModel) && hasCredentialForProvider(writerProvider);
  const judgeReady = Boolean(judgeModel) && hasCredentialForProvider(judgeProvider);

  const writerPill = document.getElementById("pill-writer");
  writerPill.textContent = `writer: ${writerReady ? "ready" : "incomplete"}`;
  writerPill.classList.toggle("ok", writerReady);
  writerPill.classList.toggle("bad", !writerReady);

  const judgePill = document.getElementById("pill-judge");
  judgePill.textContent = `judge: ${judgeReady ? "ready" : "incomplete"}`;
  judgePill.classList.toggle("ok", judgeReady);
  judgePill.classList.toggle("bad", !judgeReady);

  document.getElementById("pill-rules").textContent = `rules: ${(state.config.rules || []).length}`;

  const setupBanner = document.getElementById("setup-banner");
  if (state.meta?.load_error) {
    setupBanner.hidden = false;
    setupBanner.textContent = `Config load warning: ${state.meta.load_error}`;
    return;
  }

  const issues = [];
  if (!writerModel) issues.push("select a writer model");
  if (!judgeModel) issues.push("select a judge model");
  if (!hasCredentialForProvider(writerProvider)) issues.push(`set credentials for ${writerProvider}`);
  if (!hasCredentialForProvider(judgeProvider)) issues.push(`set credentials for ${judgeProvider}`);

  if (issues.length) {
    setupBanner.hidden = false;
    setupBanner.textContent = `Setup required: ${issues.join(", ")}.`;
  } else {
    setupBanner.hidden = true;
  }
}

function currentModelValue(role) {
  const input = document.getElementById(`${role}-model-input`);
  const select = document.getElementById(`${role}-model-select`);
  return input.classList.contains("hidden") ? select.value.trim() : input.value.trim();
}

async function fetchConfig() {
  const res = await fetch("/api/config");
  const json = await res.json();
  if (!res.ok || !json.ok) {
    throw new Error(json.error || "Failed to load config.");
  }
  state.config = json.config || defaultConfig();
  state.meta = json.meta || null;
  for (const inputId of Object.values(CREDENTIAL_INPUTS)) {
    document.getElementById(inputId).value = "";
  }
  syncFormFromState();
}

async function saveConfigToServer() {
  if (!state.config) return;
  readFormIntoState();
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.config),
  });
  const json = await res.json();
  if (!res.ok || !json.ok) {
    throw new Error(json.error || "Failed to save config.");
  }
  state.config = json.config;
  state.meta = json.meta || null;
  for (const inputId of Object.values(CREDENTIAL_INPUTS)) {
    document.getElementById(inputId).value = "";
  }
  syncFormFromState();
}

async function fetchModels(role) {
  if (!state.config) return;
  readFormIntoState();
  const res = await fetch("/api/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, settings: state.config.settings }),
  });
  const json = await res.json();
  if (!res.ok || !json.ok) {
    throw new Error(json.error || "Failed to load model list.");
  }
  state.models[role] = {
    models: Array.isArray(json.models) ? json.models : [],
    supportsListing: Boolean(json.supports_listing),
    manual: !(json.supports_listing && Array.isArray(json.models) && json.models.length > 0),
  };
  syncModelControl(role);
}

async function testConnection(role) {
  if (!state.config) throw new Error("Config not loaded.");
  readFormIntoState();
  const res = await fetch("/api/test-connection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, settings: state.config.settings }),
  });
  const json = await res.json();
  if (!res.ok || !json.ok) {
    throw new Error(json.error || "Connection test failed.");
  }
  return json.result;
}

async function runTurn(userText) {
  if (!state.config) throw new Error("Config not loaded.");
  readFormIntoState();
  const payload = {
    user_text: userText,
    thread_messages: state.chat,
    settings: state.config.settings,
    rules: state.config.rules,
    prompts: state.config.prompts,
    turn_id: state.currentTurnId,
  };

  const res = await fetch("/api/turn-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    const fallback = await fetch("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const fallbackJson = await fallback.json();
    if (!fallback.ok || !fallbackJson.ok) {
      throw new Error(fallbackJson.error || "Turn failed.");
    }
    return fallbackJson.turn;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalTurn = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let packet = null;
      try {
        packet = JSON.parse(line);
      } catch {
        continue;
      }
      if (packet.type === "event") {
        setBusy(true, statusLabelFromEvent(packet.event));
      } else if (packet.type === "turn") {
        finalTurn = packet.turn;
      } else if (packet.type === "error") {
        throw new Error(packet.error || "Turn failed.");
      }
    }
  }

  if (!finalTurn) {
    throw new Error("Turn failed: no final result returned.");
  }
  return finalTurn;
}

async function onSend() {
  if (state.busy) return;
  syncStatusPills();
  const writerProvider = document.getElementById("writer-provider").value;
  if (!currentModelValue("writer") || !hasCredentialForProvider(writerProvider)) {
    setPage("settings");
    setSettingsSection("writer");
    alert("Writer settings are incomplete. Configure the writer model and credentials first.");
    return;
  }

  const box = document.getElementById("chat-input");
  const text = box.value.trim();
  if (!text) return;

  state.chat.push({ role: "user", content: text, at: new Date().toISOString() });
  state.chat = normalizeChat(state.chat);
  saveJson(STORAGE.chat, state.chat);
  box.value = "";
  renderChat();

  try {
    state.currentTurnId = (globalThis.crypto?.randomUUID && globalThis.crypto.randomUUID()) || String(Date.now());
    setBusy(true, "starting...");
    const turn = await runTurn(text);
    state.chat.push({ role: "assistant", content: turn.final || "", at: new Date().toISOString() });
    state.chat = normalizeChat(state.chat);
    saveJson(STORAGE.chat, state.chat);
    renderChat();

    state.transcript.unshift(turn);
    saveJson(STORAGE.transcript, state.transcript);
    renderTranscript();
  } finally {
    state.currentTurnId = null;
    setBusy(false, "idle");
  }
}

async function stopTurn() {
  if (!state.currentTurnId) return;
  await fetch("/api/turn-cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ turn_id: state.currentTurnId }),
  });
}

function exportTranscript() {
  const blob = new Blob([JSON.stringify(state.transcript, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "constitutional-ai-transcript.json";
  link.click();
  URL.revokeObjectURL(url);
}

function applyProviderDefaults(role) {
  if (!state.config) return;
  const providerId = document.getElementById(`${role}-provider`).value;
  const provider = providerMeta(providerId);
  state.config.settings[role].provider = providerId;
  state.config.settings[role].model = provider.defaultModel;
  state.config.settings[role].api_base = provider.defaultApiBase || "";
  state.config.settings[role].api_version = provider.defaultApiVersion || "";
  document.getElementById(`${role}-api-base`).value = state.config.settings[role].api_base;
  document.getElementById(`${role}-api-version`).value = state.config.settings[role].api_version;
  syncEndpointFields(role);
  state.models[role] = { models: [], supportsListing: false, manual: true };
  syncModelControl(role);
  syncStatusPills();
}

function installListeners() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => setPage(tab.dataset.page));
  }
  for (const button of document.querySelectorAll(".settingsNavButton")) {
    button.addEventListener("click", () => setSettingsSection(button.dataset.settingsSection));
  }
  for (const [fieldName, inputId] of Object.entries(CREDENTIAL_INPUTS)) {
    document.getElementById(inputId).addEventListener("input", () => {
      syncCredentialSourceLabels();
      syncStatusPills();
    });
  }
  for (const role of ["writer", "judge"]) {
    document.getElementById(`${role}-provider`).addEventListener("change", () => applyProviderDefaults(role));
    document.getElementById(`${role}-api-base`).addEventListener("input", () => syncStatusPills());
    document.getElementById(`${role}-api-version`).addEventListener("input", () => syncStatusPills());
    document.getElementById(`${role}-model-input`).addEventListener("input", () => syncStatusPills());
    document.getElementById(`${role}-model-select`).addEventListener("change", () => syncStatusPills());
    document.getElementById(`btn-${role}-refresh-models`).addEventListener("click", async () => {
      const statusNode = document.getElementById(`${role}-model-status`);
      try {
        setBusy(true, "loading models...");
        await fetchModels(role);
      } catch (error) {
        state.models[role] = { models: [], supportsListing: false, manual: true };
        syncModelControl(role);
        statusNode.textContent = String(error.message || error);
      } finally {
        setBusy(false, "idle");
      }
    });
    document.getElementById(`btn-${role}-test-connection`).addEventListener("click", async () => {
      const resultNode = document.getElementById(`${role}-test-result`);
      try {
        setBusy(true, "testing connection...");
        const result = await testConnection(role);
        resultNode.textContent = `${result.message} provider=${result.provider} model=${result.model}${result.api_base ? ` api_base=${result.api_base}` : ""}`;
      } catch (error) {
        resultNode.textContent = String(error.message || error);
      } finally {
        setBusy(false, "idle");
      }
    });
  }

  for (const id of [
    "set-temperature",
    "set-max-tokens",
    "set-max-revisions",
    "set-timeout-ms",
    "set-execution-mode",
    "set-parallel-max-iterations",
    "set-max-iteration-ms",
    "rules-text",
    "prompt-writer",
    "prompt-pass",
    "prompt-critique",
  ]) {
    document.getElementById(id).addEventListener("input", () => {
      readFormIntoState();
      syncStatusPills();
    });
  }

  document.getElementById("btn-send").addEventListener("click", onSend);
  document.getElementById("btn-stop").addEventListener("click", stopTurn);
  document.getElementById("chat-input").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      onSend();
    }
  });
  document.getElementById("btn-clear-chat").addEventListener("click", () => {
    state.chat = [];
    saveJson(STORAGE.chat, state.chat);
    renderChat();
  });
  document.getElementById("btn-clear-transcript").addEventListener("click", () => {
    state.transcript = [];
    saveJson(STORAGE.transcript, state.transcript);
    renderTranscript();
  });
  document.getElementById("btn-export").addEventListener("click", exportTranscript);
  document.getElementById("btn-save-config").addEventListener("click", async () => {
    setBusy(true, "saving config...");
    try {
      await saveConfigToServer();
    } finally {
      setBusy(false, "idle");
    }
  });
  document.getElementById("btn-save-settings").addEventListener("click", async () => {
    setBusy(true, "saving settings...");
    try {
      await saveConfigToServer();
    } finally {
      setBusy(false, "idle");
    }
  });
  document.getElementById("btn-save-rules").addEventListener("click", async () => {
    setBusy(true, "saving rules...");
    try {
      await saveConfigToServer();
    } finally {
      setBusy(false, "idle");
    }
  });
  document.getElementById("btn-reload-config").addEventListener("click", async () => {
    setBusy(true, "reloading config...");
    try {
      await fetchConfig();
    } finally {
      setBusy(false, "idle");
    }
  });
  document.getElementById("btn-reset-settings").addEventListener("click", () => {
    state.config = defaultConfig();
    state.meta = state.meta || {};
    for (const inputId of Object.values(CREDENTIAL_INPUTS)) {
      document.getElementById(inputId).value = "";
    }
    state.models.writer = { models: [], supportsListing: false, manual: true };
    state.models.judge = { models: [], supportsListing: false, manual: true };
    syncFormFromState();
    setSettingsSection("credentials");
  });
  document.getElementById("btn-load-sample").addEventListener("click", () => {
    document.getElementById("rules-text").value = defaultConfig().rules.join("\n");
    readFormIntoState();
    syncStatusPills();
  });
}

async function bootstrap() {
  renderChat();
  renderTranscript();
  installListeners();
  setPage("chat");
  setSettingsSection("credentials");
  setBusy(false, "idle");
  try {
    await fetchConfig();
  } catch (error) {
    state.config = defaultConfig();
    syncFormFromState();
    alert(String(error.message || error));
  }
}

bootstrap();
