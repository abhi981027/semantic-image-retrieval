document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        if (tabName === 'text') {
            document.getElementById('textTab').classList.add('active');
            document.getElementById('queryImage').value = '';
        } else {
            document.getElementById('imageTab').classList.add('active');
            document.getElementById('queryText').value = '';
        }
    });
});

// File input label update
const updateFileLabel = (input, labelElement) => {
    input.addEventListener('change', () => {
        const fileCount = input.files.length;
        const fileTextElement = labelElement.querySelector('.file-text');
        
        if (fileCount > 0) {
            if (fileCount === 1) {
                fileTextElement.textContent = input.files[0].name;
            } else {
                fileTextElement.textContent = `${fileCount} files selected`;
            }
        } else {
            fileTextElement.textContent = input.dataset.originalText || 'Choose images or drag & drop';
        }
    });
};

const uploadInput = document.getElementById('uploadFiles');
const uploadLabel = uploadInput.nextElementSibling;
const uploadFileText = uploadLabel.querySelector('.file-text');
uploadFileText.dataset.originalText = uploadFileText.textContent;
updateFileLabel(uploadInput, uploadLabel);

const queryImageInput = document.getElementById('queryImage');
const queryImageLabel = queryImageInput.nextElementSibling;
const queryFileText = queryImageLabel.querySelector('.file-text');
queryFileText.dataset.originalText = queryFileText.textContent;
updateFileLabel(queryImageInput, queryImageLabel);

// Upload form submission
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const formData = new FormData(e.target);
    const messageDiv = document.getElementById('uploadMessage');
    const files = uploadInput.files;
    
    if (files.length === 0) {
        showMessage(messageDiv, 'Please select one or more files to upload.', 'warning');
        return;
    }

    showMessage(messageDiv, `Uploading and processing ${files.length} image(s)... Please wait.`, 'info');
    
    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showMessage(messageDiv, data.message || 'Upload complete!', 'success');
        
        // Reset form and file label
        e.target.reset();
        uploadFileText.textContent = uploadFileText.dataset.originalText;
    } catch (error) {
        showMessage(messageDiv, 'Upload failed. Please try again.', 'error');
    }
});

// Search form submission
document.getElementById('searchForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const formData = new FormData(e.target);
    const resultsDiv = document.getElementById('results');
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Show results container and loading state
    resultsContainer.style.display = 'block';
    resultsDiv.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Searching for similar images...</p>
        </div>
    `;
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const summary = document.getElementById('resultsSummary');
    summary.textContent = '';

    try {
        const response = await fetch('/search/', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            resultsDiv.innerHTML = '';
            resultsDiv.appendChild(emptyState('⚠️', data.detail || 'Search failed.'));
            return;
        }
        displayResults(data.results || []);
        if (data.timing) {
            const n = (data.results || []).length;
            summary.textContent = `${n} result${n === 1 ? '' : 's'} · search ${data.timing.search_ms.toFixed(0)} ms` +
                ` (+ ${(data.timing.model_load_ms / 1000).toFixed(1)} s model load)`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <div class="empty-state-text">Search failed. Please try again.</div>
            </div>
        `;
    }
});

// Display search results
function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (!results || results.length === 0) {
        resultsDiv.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-text">No results found. Try a different query.</div>
            </div>
        `;
        return;
    }

    results.forEach((result, index) => {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.style.animationDelay = `${index * 0.05}s`;
        
        const imageContainer = document.createElement('div');
        imageContainer.className = 'result-image-container';
        
        const img = document.createElement('img');
        img.src = result.image_url;
        img.alt = `Result ${index + 1}`;

        // Handle bounding box for object detection
        img.onload = () => {
            if (result.type === 'object' && result.box) {
                const boxDiv = document.createElement('div');
                boxDiv.className = 'detection-box';
                boxDiv.dataset.box = JSON.stringify(result.box);
                imageContainer.appendChild(boxDiv);
                placeBox(img, boxDiv);
            }
        };

        imageContainer.appendChild(img);
        if (result.timestamp) {
            const ts = document.createElement('span');
            ts.className = 'timestamp-chip';
            ts.textContent = result.timestamp;
            imageContainer.appendChild(ts);
        }
        resultCard.appendChild(imageContainer);

        const resultInfo = document.createElement('div');
        resultInfo.className = 'result-info';

        const scorePercentage = (result.score * 100).toFixed(1);
        const sourceLine = result.source_video ? `
                <div class="meta-item source-line">
                    <span class="meta-label">Source:</span>
                    <span>${escapeHtml(result.video_name)} @ ${escapeHtml(result.timestamp || '')}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Frame:</span>
                    <span>#${result.frame_number}</span>
                </div>` : '';
        resultInfo.innerHTML = `
            <div class="result-score">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="12" cy="7" r="4" stroke-width="2"/>
                </svg>
                ${scorePercentage}% Match
            </div>
            <div class="result-meta">
                ${sourceLine}
                <div class="meta-item">
                    <span class="meta-label">Type:</span>
                    <span class="type-badge ${result.type}">${result.type}</span>
                </div>
                ${result.box ? `
                    <div class="meta-item">
                        <span class="meta-label">Region:</span>
                        <span>[${result.box.map(v => Math.round(v)).join(', ')}]</span>
                    </div>
                ` : ''}
            </div>
        `;
        
        resultCard.appendChild(resultInfo);
        resultsDiv.appendChild(resultCard);
    });
}

// Place a bounding box over an image shown with object-fit: contain.
// The image may be letterboxed inside its container, so add the offset of the drawn area.
function placeBox(img, boxDiv) {
    const box = JSON.parse(boxDiv.dataset.box);
    const cw = img.clientWidth, ch = img.clientHeight;
    const nw = img.naturalWidth, nh = img.naturalHeight;
    if (!nw || !nh) return;
    const scale = Math.min(cw / nw, ch / nh);
    const offX = (cw - nw * scale) / 2;
    const offY = (ch - nh * scale) / 2;
    boxDiv.style.left = (offX + box[0] * scale) + 'px';
    boxDiv.style.top = (offY + box[1] * scale) + 'px';
    boxDiv.style.width = ((box[2] - box[0]) * scale) + 'px';
    boxDiv.style.height = ((box[3] - box[1]) * scale) + 'px';
}

window.addEventListener('resize', () => {
    document.querySelectorAll('.detection-box').forEach(boxDiv => {
        const img = boxDiv.parentElement.querySelector('img');
        if (img) placeBox(img, boxDiv);
    });
});

function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function emptyState(icon, text) {
    const div = document.createElement('div');
    div.className = 'empty-state';
    div.innerHTML = `<div class="empty-state-icon">${icon}</div><div class="empty-state-text"></div>`;
    div.querySelector('.empty-state-text').textContent = text;
    return div;
}

// ---------------------------------------------------------------------------
// Video upload + progress polling
// ---------------------------------------------------------------------------
const videoInput = document.getElementById('videoFile');
const videoLabel = videoInput.nextElementSibling;
const videoFileText = videoLabel.querySelector('.file-text');
videoFileText.dataset.originalText = videoFileText.textContent;
updateFileLabel(videoInput, videoLabel);

const STAGE_LABELS = {
    queued: 'Queued',
    waiting: 'Waiting for another ingestion to finish',
    loading_models: 'Loading SAM + CLIP',
    extracting: 'Extracting keyframes',
    ingesting: 'Running SAM + CLIP on keyframes',
    done: 'Done',
    error: 'Failed',
    skipped: 'Already indexed',
};

function fmtDuration(sec) {
    sec = Math.round(sec || 0);
    const m = Math.floor(sec / 60), s = sec % 60;
    return m ? `${m}m ${s}s` : `${s}s`;
}

function renderJob(job) {
    const box = document.getElementById('videoProgress');
    const label = document.getElementById('videoProgressLabel');
    const fill = document.getElementById('videoProgressFill');
    const detail = document.getElementById('videoProgressDetail');
    box.hidden = false;
    box.dataset.stage = job.stage;

    const stage = STAGE_LABELS[job.stage] || job.stage;
    label.textContent = `${job.video_name || ''} · ${stage}`;

    let pct = 0;
    if (job.stage === 'extracting' && job.total) pct = 30 * job.done / job.total;
    else if (job.stage === 'ingesting' && job.total) pct = 30 + 70 * job.done / job.total;
    else if (job.stage === 'done' || job.stage === 'skipped') pct = 100;
    fill.style.width = pct.toFixed(1) + '%';

    if (job.stage === 'extracting') {
        detail.textContent = `Sampled ${job.done} / ~${job.total} frames · ${fmtDuration(job.elapsed_sec)} elapsed`;
    } else if (job.stage === 'ingesting') {
        detail.textContent = `Keyframe ${job.done} / ${job.total} · ${fmtDuration(job.elapsed_sec)} elapsed`;
    } else if (job.stage === 'done') {
        detail.textContent = `${job.message} Extraction ${job.extraction_sec.toFixed(1)} s, ` +
            `SAM+CLIP ${fmtDuration(job.ingestion_sec)} (${job.mean_sec_per_keyframe.toFixed(1)} s per keyframe).`;
    } else {
        detail.textContent = job.message || '';
    }
}

async function pollJob(jobId) {
    while (true) {
        await new Promise(r => setTimeout(r, 2000));
        let job;
        try {
            const res = await fetch(`/video-status/${jobId}`);
            job = await res.json();
        } catch (e) {
            continue; // server busy (e.g. reloading); try again
        }
        renderJob(job);
        if (job.stage === 'done' || job.stage === 'error') {
            if (job.stage === 'done') loadVideoList();
            return job;
        }
    }
}

document.getElementById('videoForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    e.stopPropagation();
    if (videoInput.files.length === 0) return;

    const submitBtn = document.getElementById('videoSubmit');
    submitBtn.disabled = true;
    renderJob({ video_name: videoInput.files[0].name, stage: 'queued', message: 'Uploading video…' });

    try {
        const response = await fetch('/upload-video/', { method: 'POST', body: new FormData(e.target) });
        const job = await response.json();
        if (!response.ok) {
            renderJob({ video_name: videoInput.files[0].name, stage: 'error', message: job.detail || 'Upload failed.' });
            return;
        }
        renderJob(job);
        e.target.reset();
        videoFileText.textContent = videoFileText.dataset.originalText;
        if (job.job_id) await pollJob(job.job_id);
    } catch (error) {
        renderJob({ video_name: '', stage: 'error', message: 'Upload failed. Please try again.' });
    } finally {
        submitBtn.disabled = false;
    }
});

// Populate the "Search in" dropdown with indexed videos
async function loadVideoList() {
    const select = document.getElementById('sourceFilter');
    try {
        const res = await fetch('/videos/');
        const data = await res.json();
        const current = select.value;
        select.querySelectorAll('option[data-video]').forEach(o => o.remove());
        (data.videos || []).forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.source_video;
            opt.dataset.video = '1';
            opt.textContent = `${v.name} (${v.keyframes} keyframes)`;
            select.appendChild(opt);
        });
        select.value = [...select.options].some(o => o.value === current) ? current : '';
    } catch (e) { /* ignore: list is optional */ }
}
loadVideoList();

// Show message helper
function showMessage(element, text, type = 'info') {
    element.textContent = text;
    element.className = 'message show';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        element.classList.remove('show');
    }, 5000);
}

// Drag and drop functionality
const setupDragAndDrop = (input, label) => {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        label.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        label.addEventListener(eventName, () => {
            label.style.background = 'rgba(255, 255, 255, 0.15)';
            label.style.borderColor = 'rgba(255, 255, 255, 0.5)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        label.addEventListener(eventName, () => {
            label.style.background = 'rgba(255, 255, 255, 0.05)';
            label.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        }, false);
    });

    label.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        input.files = files;
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        input.dispatchEvent(event);
    }, false);
};

setupDragAndDrop(uploadInput, uploadLabel);
setupDragAndDrop(queryImageInput, queryImageLabel);
setupDragAndDrop(videoInput, videoLabel);
