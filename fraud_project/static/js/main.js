/**
 * FraudGuard AI - Client-side interactivity
 * Handles Drag & Drop, File Validation Preview, Real-time Table Filter, and Risk Selection.
 */

document.addEventListener("DOMContentLoaded", () => {
    initDragAndDrop();
    initTableFilters();
    initSubmitLoading();
});

/**
 * Configure drag and drop mechanics for CSV upload box.
 */
function initDragAndDrop() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const fileInfoBadge = document.getElementById("fileInfoBadge");
    const fileNameDisplay = document.getElementById("fileNameDisplay");

    if (!dropZone || !fileInput) return;

    // Trigger file dialog on drop zone click
    dropZone.addEventListener("click", () => fileInput.click());

    // Prevent default drag behaviors
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    // Highlight drop zone on drag over
    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"), false);
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"), false);
    });

    // Handle dropped files
    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            displaySelectedFile(files[0]);
        }
    });

    // Handle standard file selection
    fileInput.addEventListener("change", (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            displaySelectedFile(fileInput.files[0]);
        }
    });

    function displaySelectedFile(file) {
        if (!file.name.toLowerCase().endsWith(".csv")) {
            alert("Please choose a valid .CSV file.");
            fileInput.value = "";
            if (fileInfoBadge) fileInfoBadge.style.display = "none";
            return;
        }

        const sizeKb = (file.size / 1024).toFixed(1);
        if (fileNameDisplay) {
            fileNameDisplay.textContent = `${file.name} (${sizeKb} KB)`;
        }
        if (fileInfoBadge) {
            fileInfoBadge.style.display = "inline-block";
        }
    }
}

/**
 * Client-side search and risk level filtering for the results table.
 */
function initTableFilters() {
    const searchInput = document.getElementById("tableSearch");
    const riskFilter = document.getElementById("riskFilter");
    const table = document.getElementById("auditTable");

    if (!table) return;

    const rows = table.querySelectorAll("tbody tr");

    function filterRows() {
        const query = searchInput ? searchInput.value.toLowerCase().trim() : "";
        const selectedRisk = riskFilter ? riskFilter.value.toUpperCase() : "ALL";

        rows.forEach((row) => {
            const textContent = row.textContent.toLowerCase();
            const rowRisk = (row.getAttribute("data-risk") || "").toUpperCase();

            const matchesQuery = query === "" || textContent.includes(query);
            const matchesRisk = selectedRisk === "ALL" || rowRisk === selectedRisk;

            if (matchesQuery && matchesRisk) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }
        });
    }

    if (searchInput) {
        searchInput.addEventListener("input", filterRows);
    }
    if (riskFilter) {
        riskFilter.addEventListener("change", filterRows);
    }
}

/**
 * Show loading spinner / state when submitting transaction batch.
 */
function initSubmitLoading() {
    const form = document.getElementById("uploadForm");
    const btn = document.getElementById("detectBtn");

    if (!form || !btn) return;

    form.addEventListener("submit", () => {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Analyzing Transactions...</span>';
    });
}
