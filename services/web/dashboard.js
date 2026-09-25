const PAGE_SIZE = 25;
const REFRESH_MS = 5000;
const REQUEST_TIMEOUT_MS = 8000;

let currentPage = 1;
let totalPages = 0;
let refreshInFlight = false;

const elements = {
    connectionState: document.getElementById("connectionState"),
    errorBanner: document.getElementById("errorBanner"),
    errorMessage: document.getElementById("errorMessage"),
    totalJobs: document.getElementById("totalJobs"),
    pageJobs: document.getElementById("pageJobs"),
    latestStatus: document.getElementById("latestStatus"),
    lastUpdated: document.getElementById("lastUpdated"),
    jobsBody: document.getElementById("jobsBody"),
    previousButton: document.getElementById("previousButton"),
    nextButton: document.getElementById("nextButton"),
    pageLabel: document.getElementById("pageLabel"),
    refreshButton: document.getElementById("refreshButton"),
};

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatDate(value) {
    if (!value) {
        return "N/A";
    }

    const parsed = new Date(value);

    if (Number.isNaN(parsed.getTime())) {
        return "N/A";
    }

    return parsed.toLocaleString();
}

function statusClass(status) {
    const normalized = String(status || "unknown").toLowerCase();

    if (
        normalized === "succeeded" ||
        normalized === "running" ||
        normalized === "queued" ||
        normalized === "failed"
    ) {
        return `status-${normalized}`;
    }

    return "status-unknown";
}

function renderRows(items) {
    if (items.length === 0) {
        elements.jobsBody.innerHTML = `
            <tr>
                <td colspan="7" class="empty">
                    No ingestion jobs have been recorded.
                </td>
            </tr>
        `;
        return;
    }

    elements.jobsBody.innerHTML = items
        .map((item) => {
            const status = escapeHtml(item.status);
            const sourceId = escapeHtml(item.source_id);
            const model = escapeHtml(item.model_name);
            const modelVersion = escapeHtml(item.model_version);

            return `
                <tr>
                    <td>#${escapeHtml(item.id)}</td>
                    <td>
                        <span class="status ${statusClass(item.status)}">
                            ${status}
                        </span>
                    </td>
                    <td>${sourceId}</td>
                    <td>${escapeHtml(item.frame_count)}</td>
                    <td>${escapeHtml(item.detection_count)}</td>
                    <td title="${modelVersion}">
                        ${model}
                    </td>
                    <td>${escapeHtml(formatDate(item.created_at))}</td>
                </tr>
            `;
        })
        .join("");
}

function setConnection(ok, message = "") {
    elements.connectionState.textContent = ok
        ? "API connected"
        : "API unavailable";

    elements.connectionState.className = ok
        ? "state state-ok"
        : "state state-error";

    if (ok) {
        elements.errorBanner.hidden = true;
        return;
    }

    elements.errorMessage.textContent =
        message || "Job data could not be refreshed.";

    elements.errorBanner.hidden = false;
}

function setLoadingState() {
    if (elements.totalJobs.textContent === "N/A") {
        elements.jobsBody.innerHTML = `
            <tr>
                <td colspan="7" class="empty">
                    Loading real job data...
                </td>
            </tr>
        `;
    }
}

async function fetchJobs() {
    if (refreshInFlight) {
        return;
    }

    refreshInFlight = true;
    setLoadingState();

    const controller = new AbortController();
    const timeout = window.setTimeout(
        () => controller.abort(),
        REQUEST_TIMEOUT_MS,
    );

    try {
        let body = null;

        for (let attempt = 0; attempt < 2; attempt += 1) {
            const response = await fetch(
                `/ingestion/jobs?page=${currentPage}&page_size=${PAGE_SIZE}`,
                {
                    method: "GET",
                    headers: {
                        Accept: "application/json",
                    },
                    cache: "no-store",
                    signal: controller.signal,
                },
            );

            if (!response.ok) {
                let detail = `HTTP ${response.status}`;

                try {
                    const responseBody = await response.json();

                    if (
                        responseBody &&
                        typeof responseBody.detail === "string"
                    ) {
                        detail = responseBody.detail;
                    }
                } catch {
                    // Keep the HTTP status as the error detail.
                }

                throw new Error(detail);
            }

            body = await response.json();

            totalPages = Number(body.pages) || 0;

            if (totalPages === 0 && currentPage > 1) {
                currentPage = 1;
                continue;
            }

            if (
                totalPages > 0 &&
                currentPage > totalPages
            ) {
                currentPage = totalPages;
                continue;
            }

            break;
        }

        if (!body) {
            throw new Error("Job data could not be loaded.");
        }

        const items = Array.isArray(body.items)
            ? body.items
            : [];

        elements.totalJobs.textContent = String(
            body.total ?? 0,
        );

        elements.pageJobs.textContent = String(
            items.length,
        );

        elements.latestStatus.textContent = items.length
            ? String(items[0].status)
            : "N/A";

        elements.pageLabel.textContent =
            `Page ${currentPage} of ${totalPages}`;

        elements.previousButton.disabled =
            currentPage <= 1;

        elements.nextButton.disabled =
            totalPages === 0 ||
            currentPage >= totalPages;

        renderRows(items);

        elements.lastUpdated.textContent =
            `Last updated ${new Date().toLocaleTimeString()}`;

        setConnection(true);
    } catch (error) {
        const message =
            error instanceof DOMException &&
            error.name === "AbortError"
                ? "Request timed out."
                : error instanceof Error
                    ? error.message
                    : "Unknown API error.";

        setConnection(false, message);
    } finally {
        window.clearTimeout(timeout);
        refreshInFlight = false;
    }
}

elements.refreshButton.addEventListener(
    "click",
    () => {
        fetchJobs();
    },
);

elements.previousButton.addEventListener(
    "click",
    () => {
        if (currentPage <= 1) {
            return;
        }

        currentPage -= 1;
        fetchJobs();
    },
);

elements.nextButton.addEventListener(
    "click",
    () => {
        if (
            totalPages === 0 ||
            currentPage >= totalPages
        ) {
            return;
        }

        currentPage += 1;
        fetchJobs();
    },
);

fetchJobs();
window.setInterval(fetchJobs, REFRESH_MS);
