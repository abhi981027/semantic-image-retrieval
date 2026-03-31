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

    try {
        const response = await fetch('/search/', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayResults(data.results || []);
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
                
                // Calculate scaling factor
                const scaleX = img.width / img.naturalWidth;
                const scaleY = img.height / img.naturalHeight;
                
                // Apply scaled coordinates
                const left = result.box[0] * scaleX;
                const top = result.box[1] * scaleY;
                const width = (result.box[2] - result.box[0]) * scaleX;
                const height = (result.box[3] - result.box[1]) * scaleY;
                
                boxDiv.style.left = left + 'px';
                boxDiv.style.top = top + 'px';
                boxDiv.style.width = width + 'px';
                boxDiv.style.height = height + 'px';
                
                imageContainer.appendChild(boxDiv);
            }
        };
        
        imageContainer.appendChild(img);
        resultCard.appendChild(imageContainer);
        
        const resultInfo = document.createElement('div');
        resultInfo.className = 'result-info';
        
        const scorePercentage = (result.score * 100).toFixed(1);
        resultInfo.innerHTML = `
            <div class="result-score">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="12" cy="7" r="4" stroke-width="2"/>
                </svg>
                ${scorePercentage}% Match
            </div>
            <div class="result-meta">
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
