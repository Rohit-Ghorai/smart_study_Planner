document.addEventListener("DOMContentLoaded", () => {
  const generateForm = document.getElementById("generate-form");
  const subjectsList = document.getElementById("subjects-list");
  const subjectCountInput = document.getElementById("subject_count");
  const addButton = document.getElementById("add-subject");
  const presetButtons = document.querySelectorAll(".preset-button");
  const resultsSection = document.getElementById("results-section");

  const generationCard = document.getElementById("generation-progress");
  const generationFill = document.getElementById("generation-progress-fill");
  const generationPercent = document.getElementById("generation-percent");
  const generationText = document.getElementById("generation-text");

  const episodesInput = document.getElementById("episodes");
  const alphaInput = document.getElementById("alpha");
  const gammaInput = document.getElementById("gamma");
  const epsilonInput = document.getElementById("epsilon");
  const epsilonDecayInput = document.getElementById("epsilon_decay");
  const minEpsilonInput = document.getElementById("min_epsilon");
  const notificationList = document.getElementById("notification-list");
  const notificationBadge = document.getElementById("notification-badge");
  const notificationPopover = document.getElementById("notification-popover");
  const notificationPopoverList = document.getElementById("notification-popover-list");
  const notificationWrap = document.getElementById("notification-wrap");
  let notificationsLoaded = false;

  function extractSubject(actionName) {
    if (!actionName) {
      return "Free";
    }
    if (["Break", "Free", "Meditation", "Snack", "Water_Break"].includes(actionName)) {
      return actionName;
    }
    return actionName.includes("_") ? actionName.split("_")[0] : actionName;
  }

  function updateGenerationProgress(value, text) {
    const bounded = Math.max(0, Math.min(100, value));
    if (generationFill) {
      generationFill.style.width = `${bounded}%`;
      generationFill.setAttribute("aria-valuenow", String(Math.round(bounded)));
    }
    if (generationPercent) {
      generationPercent.textContent = `${Math.round(bounded)}%`;
    }
    if (generationText && text) {
      generationText.textContent = text;
    }
  }

  function syncNames() {
    if (!subjectsList || !subjectCountInput) {
      return;
    }

    const rows = Array.from(subjectsList.querySelectorAll(".subject-row"));
    subjectCountInput.value = rows.length;

    rows.forEach((row, index) => {
      const inputs = row.querySelectorAll("input");
      inputs[0].name = `subject_name_${index}`;
      inputs[1].name = `difficulty_${index}`;
      inputs[2].name = `strength_${index}`;
    });
  }

  function createRow() {
    if (!subjectsList) {
      return;
    }

    const row = document.createElement("div");
    row.className = "subject-row";

    row.innerHTML = `
      <div class="field-row grow">
        <label>Subject name</label>
        <input type="text" placeholder="e.g. OS" value="">
      </div>
      <div class="field-row small">
        <label>Difficulty</label>
        <input type="number" min="1" max="5" value="3">
      </div>
      <div class="field-row small">
        <label>Confidence</label>
        <input type="number" min="1" max="5" value="3">
      </div>
      <button type="button" class="remove-subject" aria-label="Remove subject">×</button>
    `;

    const removeButton = row.querySelector(".remove-subject");
    removeButton.addEventListener("click", () => {
      if (subjectsList.querySelectorAll(".subject-row").length > 1) {
        row.remove();
        syncNames();
      }
    });

    subjectsList.appendChild(row);
    syncNames();
  }

  function collectSubjectContext() {
    const rows = Array.from(document.querySelectorAll("#subjects-list .subject-row"));
    const subjects = [];
    const difficulties = {};
    const strengths = {};

    rows.forEach((row) => {
      const inputs = row.querySelectorAll("input");
      if (inputs.length < 3) {
        return;
      }

      const subject = inputs[0].value.trim();
      const difficulty = Number.parseInt(inputs[1].value || "3", 10);
      const strength = Number.parseInt(inputs[2].value || "3", 10);
      if (!subject) {
        return;
      }
      subjects.push(subject);
      difficulties[subject] = Number.isNaN(difficulty) ? 3 : difficulty;
      strengths[subject] = Number.isNaN(strength) ? 3 : strength;
    });

    return { subjects, difficulties, strengths };
  }

  function renderNotificationItems(target, notifications, emptyMessage) {
    if (!target) {
      return;
    }
    if (!notifications.length) {
      target.innerHTML = `<p class='muted-copy'>${emptyMessage}</p>`;
      return;
    }

    target.innerHTML = notifications
      .slice(0, 6)
      .map(
        (item) => `
          <button type="button" class="notification-item ${item.is_read ? "is-read" : "is-unread"} ${item.kind || "info"}" data-notification-id="${item.id}">
            <strong>${String(item.kind || "update").replace(/-/g, " ")}</strong>
            <p>${item.message}</p>
            <span class="notification-hint">${item.is_read ? "Read" : "Tap to mark read"}</span>
          </button>
        `,
      )
      .join("");
  }

  async function refreshNotifications() {
    if (!notificationList && !notificationPopoverList && !notificationBadge) {
      return;
    }

    try {
      const response = await fetch("/api/notifications", { headers: { "X-Requested-With": "XMLHttpRequest" } });
      if (!response.ok) {
        throw new Error(`Notification fetch failed: ${response.status}`);
      }

      const payload = await response.json();
      const notifications = payload.notifications || [];
      renderNotificationItems(notificationList, notifications, "No notifications yet.");
      renderNotificationItems(notificationPopoverList, notifications, "No notifications yet.");

      if (notificationBadge) {
        const unreadCount = payload.unread_count || 0;
        if (unreadCount > 0) {
          notificationBadge.textContent = `${unreadCount} alerts`;
          notificationBadge.hidden = false;
        } else {
          notificationBadge.hidden = true;
          if (notificationPopover) {
            notificationPopover.hidden = true;
            notificationBadge.setAttribute("aria-expanded", "false");
          }
        }
      }

      notificationsLoaded = true;
    } catch (error) {
      renderNotificationItems(notificationList, [], "Could not load notifications.");
      renderNotificationItems(notificationPopoverList, [], "Could not load notifications.");
      console.error(error);
    }
  }

  async function markNotificationRead(notificationId) {
    const response = await fetch(`/api/notifications/${notificationId}/read`, {
      method: "POST",
      headers: {
        "X-Requested-With": "XMLHttpRequest",
      },
    });

    if (!response.ok) {
      throw new Error(`Mark read failed: ${response.status}`);
    }

    await refreshNotifications();
  }

  async function toggleNotificationPopover(forceOpen = null) {
    if (!notificationPopover || !notificationBadge) {
      return;
    }

    const shouldOpen = forceOpen === null ? notificationPopover.hidden : Boolean(forceOpen);
    notificationPopover.hidden = !shouldOpen;
    notificationBadge.setAttribute("aria-expanded", shouldOpen ? "true" : "false");

    if (shouldOpen && !notificationsLoaded) {
      await refreshNotifications();
    }
  }

  function initializeProgressTracker() {
    const trackerList = document.getElementById("tracker-list");
    const trackerContainer = document.getElementById("progress-tracker");
    const replanNote = document.getElementById("replan-note");

    if (!trackerList || !trackerContainer) {
      return;
    }

    trackerList.addEventListener("click", async (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }

      const trackerRow = target.closest(".tracker-row");
      if (!trackerRow) {
        return;
      }

      const slotIndex = Number.parseInt(trackerRow.dataset.slotIndex || "-1", 10);
      const timeSlot = trackerRow.dataset.timeSlot || "";
      const action = trackerRow.dataset.action || "";
      const planIdValue = trackerContainer.dataset.planId || "";
      const planId = planIdValue ? Number.parseInt(planIdValue, 10) : null;

      if (target.classList.contains("progress-btn")) {
        const status = target.dataset.status || "completed";
        const subject = extractSubject(action);
        try {
          const response = await fetch("/api/progress-event", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Requested-With": "XMLHttpRequest",
            },
            body: JSON.stringify({
              plan_id: Number.isNaN(planId) ? null : planId,
              subject,
              time_slot: timeSlot,
              status,
              scheduled_date: new Date().toISOString().slice(0, 10),
            }),
          });

          if (!response.ok) {
            throw new Error(`Progress save failed: ${response.status}`);
          }

          trackerRow.classList.remove("is-completed", "is-missed");
          trackerRow.classList.add(status === "completed" ? "is-completed" : "is-missed");
          if (replanNote) {
            replanNote.textContent =
              status === "completed"
                ? `${timeSlot} marked completed.`
                : `${timeSlot} marked missed. Click Adaptive Replan to rebalance remaining slots.`;
          }
          await refreshNotifications();
        } catch (error) {
          if (replanNote) {
            replanNote.textContent = "Could not save progress. Please try again.";
          }
          console.error(error);
        }
      }

      if (target.classList.contains("replan-btn") && slotIndex >= 0) {
        const schedule = Array.from(document.querySelectorAll(".optimized-action")).map((element) =>
          element.textContent.trim(),
        );
        const subjectContext = collectSubjectContext();
        const examDateInput = document.getElementById("exam_date");

        try {
          const response = await fetch("/api/adaptive-replan", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Requested-With": "XMLHttpRequest",
            },
            body: JSON.stringify({
              schedule,
              missed_slot_index: slotIndex,
              subjects: subjectContext.subjects,
              difficulties: subjectContext.difficulties,
              strengths: subjectContext.strengths,
              exam_date: examDateInput?.value || "",
            }),
          });

          if (!response.ok) {
            throw new Error(`Adaptive replan failed: ${response.status}`);
          }

          const payload = await response.json();
          const updated = payload.schedule || schedule;

          document.querySelectorAll(".optimized-action").forEach((element, index) => {
            if (updated[index]) {
              element.textContent = updated[index];
            }
          });

          document.querySelectorAll(".tracker-row").forEach((row, index) => {
            if (updated[index]) {
              row.dataset.action = updated[index];
              const actionElement = row.querySelector(".tracker-action");
              if (actionElement) {
                actionElement.textContent = updated[index];
              }
            }
          });

          if (replanNote) {
            replanNote.textContent = payload.note || "Adaptive replan generated.";
          }
        } catch (error) {
          if (replanNote) {
            replanNote.textContent = "Adaptive replan failed. Please retry.";
          }
          console.error(error);
        }
      }
    });
  }

  notificationBadge?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    void toggleNotificationPopover();
  });

  notificationPopoverList?.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const item = target.closest(".notification-item");
    if (!item) {
      return;
    }

    const notificationId = Number.parseInt(item.dataset.notificationId || "0", 10);
    if (!notificationId || item.classList.contains("is-read")) {
      return;
    }

    try {
      await markNotificationRead(notificationId);
    } catch (error) {
      console.error(error);
    }
  });

  document.addEventListener("click", (event) => {
    if (!notificationWrap || !notificationPopover || !notificationBadge) {
      return;
    }

    const target = event.target;
    if (!(target instanceof Node)) {
      return;
    }

    if (notificationWrap.contains(target)) {
      return;
    }

    if (!notificationPopover.hidden) {
      notificationPopover.hidden = true;
      notificationBadge.setAttribute("aria-expanded", "false");
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && notificationPopover && notificationBadge) {
      notificationPopover.hidden = true;
      notificationBadge.setAttribute("aria-expanded", "false");
    }
  });

  function initializeNotificationSettings() {
    const settingsRoot = document.getElementById("notify-settings");
    const statusElement = document.getElementById("notify-settings-status");
    if (!settingsRoot) {
      return;
    }

    settingsRoot.addEventListener("change", async (event) => {
      const target = event.target;
      if (!(target instanceof HTMLInputElement) || !target.classList.contains("notify-setting")) {
        return;
      }

      const payload = {
        in_app_enabled: Boolean(settingsRoot.querySelector('[data-key="in_app_enabled"]')?.checked),
        daily_reminder_enabled: Boolean(settingsRoot.querySelector('[data-key="daily_reminder_enabled"]')?.checked),
        streak_alert_enabled: Boolean(settingsRoot.querySelector('[data-key="streak_alert_enabled"]')?.checked),
        missed_slot_alert_enabled: Boolean(settingsRoot.querySelector('[data-key="missed_slot_alert_enabled"]')?.checked),
      };

      try {
        const response = await fetch("/api/notification-settings", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
          },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`Notification settings save failed: ${response.status}`);
        }
        if (statusElement) {
          statusElement.textContent = "Settings saved.";
        }
      } catch (error) {
        if (statusElement) {
          statusElement.textContent = "Could not save settings. Please retry.";
        }
        console.error(error);
      }
    });
  }

  function initializeDynamicProgressBars() {
    document.querySelectorAll(".dynamic-progress").forEach((element) => {
      const rate = Number.parseFloat(element.dataset.rate || "0");
      const bounded = Number.isNaN(rate) ? 0 : Math.max(0, Math.min(100, rate));
      element.style.width = `${bounded}%`;
    });

    document.querySelectorAll(".dynamic-trend-bar").forEach((element) => {
      const value = Number.parseInt(element.dataset.value || "0", 10);
      const bounded = Number.isNaN(value) ? 0 : Math.max(0, Math.min(5, value));
      const height = bounded === 0 ? 12 : Math.min(58, 14 + bounded * 10);
      element.style.height = `${height}px`;
    });
  }

  addButton?.addEventListener("click", () => createRow());

  presetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const presetName = button.dataset.preset;

      if (presetName === "fast") {
        episodesInput.value = "400";
        alphaInput.value = "0.12";
        gammaInput.value = "0.85";
        epsilonInput.value = "1.0";
        epsilonDecayInput.value = "0.985";
        minEpsilonInput.value = "0.08";
      }

      if (presetName === "balanced") {
        episodesInput.value = "1500";
        alphaInput.value = "0.10";
        gammaInput.value = "0.90";
        epsilonInput.value = "1.0";
        epsilonDecayInput.value = "0.995";
        minEpsilonInput.value = "0.05";
      }

      if (presetName === "deep") {
        episodesInput.value = "3500";
        alphaInput.value = "0.08";
        gammaInput.value = "0.95";
        epsilonInput.value = "1.0";
        epsilonDecayInput.value = "0.997";
        minEpsilonInput.value = "0.03";
      }

      presetButtons.forEach((element) => element.classList.remove("active"));
      button.classList.add("active");
    });
  });

  subjectsList?.querySelectorAll(".subject-row").forEach((row) => {
    row.querySelector(".remove-subject")?.addEventListener("click", () => {
      if (subjectsList.querySelectorAll(".subject-row").length > 1) {
        row.remove();
        syncNames();
      }
    });
  });

  generateForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    syncNames();

    const submitButton = generateForm.querySelector('button[type="submit"]');
    const originalButtonText = submitButton?.textContent || "Generate study plan";
    if (submitButton) {
      submitButton.disabled = true;
      submitButton.textContent = "Generating...";
    }

    if (generationCard) {
      generationCard.hidden = false;
    }
    updateGenerationProgress(8, "Preparing input data...");
    generationCard?.scrollIntoView({ behavior: "smooth", block: "start" });

    const milestones = [
      { cap: 22, text: "Building RL environment..." },
      { cap: 46, text: "Training Q-Learning and SARSA..." },
      { cap: 70, text: "Evaluating policy and rewards..." },
      { cap: 90, text: "Formatting plan and insights..." },
    ];

    let progress = 8;
    let milestoneIndex = 0;
    let progressInterval = null;

    progressInterval = window.setInterval(() => {
      const current = milestones[Math.min(milestoneIndex, milestones.length - 1)];
      if (progress < current.cap) {
        progress += Math.max(1, Math.ceil((current.cap - progress) / 6));
        updateGenerationProgress(progress, current.text);
      } else if (milestoneIndex < milestones.length - 1) {
        milestoneIndex += 1;
      }
    }, 350);

    try {
      const response = await fetch(generateForm.action, {
        method: "POST",
        body: new FormData(generateForm),
        headers: {
          "X-Requested-With": "XMLHttpRequest",
        },
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const html = await response.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, "text/html");
      const incomingResults = doc.getElementById("results-section");

      if (!incomingResults || !resultsSection) {
        throw new Error("Could not extract updated results section");
      }

      updateGenerationProgress(100, "Plan ready. Showing results...");
      resultsSection.innerHTML = incomingResults.innerHTML;
      initializeProgressTracker();
      await refreshNotifications();

      setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 180);
    } catch (error) {
      updateGenerationProgress(100, "Could not generate plan. Please try again.");
      if (resultsSection) {
        resultsSection.innerHTML = `
          <section class="card status-card error">
            <strong>Request failed</strong>
            <ul><li>Plan generation failed. Please try again.</li></ul>
          </section>
        `;
      }
      console.error(error);
    } finally {
      if (progressInterval) {
        window.clearInterval(progressInterval);
      }
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.textContent = originalButtonText;
      }
    }
  });

  syncNames();
  initializeProgressTracker();
  initializeNotificationSettings();
  initializeDynamicProgressBars();
  void refreshNotifications();
});