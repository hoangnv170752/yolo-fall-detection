// Global variables
const apiBaseUrl = window.location.origin;
const files = [];
let benchmarkResults = [];
let processingTimeChart = null;
let confidenceChart = null;
let historyResults = [];

// DOM elements
document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const processBtn = document.getElementById('process-btn');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const historyGallery = document.getElementById('history-gallery');
    const randomPhotoBtn = document.getElementById('random-photo-btn');
    
    // Initialize event listeners
    initDragAndDrop();
    initFileInput();
    initTabSwitching();
    initProcessButton();
    initHistoryTab();
    initRandomPhotoButton();
    
    // Initialize process button
    function initProcessButton() {
        processBtn.addEventListener('click', processFiles);
    }
    
    // Create placeholder for upload icon
    createUploadIcon();
    
    // Initialize random photo button
    function initRandomPhotoButton() {
        if (randomPhotoBtn) {
            randomPhotoBtn.addEventListener('click', fetchRandomPhoto);
        }
    }
    
    // Fetch a random photo from the dataset
    async function fetchRandomPhoto() {
        try {
            // Show loading state
            randomPhotoBtn.disabled = true;
            randomPhotoBtn.textContent = 'Loading...';
            
            // Fetch random photo from API
            const response = await fetch(`${apiBaseUrl}/random-photo`);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server returned ${response.status}`);
            }
            
            // Get filename from header
            const filename = response.headers.get('X-Image-Filename') || 'random_photo.jpg';
            
            // Convert response to blob
            const blob = await response.blob();
            
            // Create a File object from the blob
            const file = new File([blob], filename, { type: blob.type });
            
            // Add file to the list
            handleFiles([file]);
            
            // Show success notification
            showNotification(`Random photo '${filename}' imported successfully`, 'success');
        } catch (error) {
            console.error('Error fetching random photo:', error);
            showNotification(`Error: ${error.message}`, 'error');
        } finally {
            // Reset button state
            randomPhotoBtn.disabled = false;
            randomPhotoBtn.textContent = 'Import Random Photo';
        }
    }
    
    // Initialize charts
    initCharts();
    
    // Drag and drop functionality
    function initDragAndDrop() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const newFiles = dt.files;
            handleFiles(newFiles);
        }
    }
    
    // File input handling
    function initFileInput() {
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
    }
    
    // Process files when added
    function handleFiles(newFiles) {
        if (newFiles.length === 0) return;
        
        Array.from(newFiles).forEach(file => {
            // Check if file is image or video
            if (!file.type.match('image.*') && !file.type.match('video.*')) {
                showNotification('Only image and video files are supported', 'error');
                return;
            }
            
            // Add file to array if not already added
            if (!files.some(f => f.name === file.name && f.size === file.size)) {
                files.push(file);
                addFileToList(file);
            }
        });
        
        // Enable process button if files are added
        if (files.length > 0) {
            processBtn.disabled = false;
        }
    }
    
    // Add file to the list in UI
    function addFileToList(file) {
        const li = document.createElement('li');
        li.className = 'file-item';
        
        const fileType = file.type.startsWith('image/') ? 'image' : 'video';
        const fileIcon = fileType === 'image' ? '🖼️' : '🎬';
        
        li.innerHTML = `
            <div class="file-info">
                <span class="file-icon">${fileIcon}</span>
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
            <button class="remove-file-btn" data-name="${file.name}" data-size="${file.size}">✕</button>
        `;
        
        // Add remove button functionality
        li.querySelector('.remove-file-btn').addEventListener('click', function() {
            const fileName = this.getAttribute('data-name');
            const fileSize = parseInt(this.getAttribute('data-size'));
            removeFile(fileName, fileSize);
            li.remove();
            
            // Disable process button if no files are left
            if (files.length === 0) {
                processBtn.disabled = true;
            }
        });
        
        fileList.appendChild(li);
    }
    
    // Remove file from array
    function removeFile(fileName, fileSize) {
        const index = files.findIndex(f => f.name === fileName && f.size === fileSize);
        if (index !== -1) {
            files.splice(index, 1);
        }
    }
    
    // Format file size for display
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Tab switching functionality
    function initTabSwitching() {
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.getAttribute('data-tab');
                
                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Update active tab content
                tabContents.forEach(content => content.classList.remove('active'));
                document.getElementById(`${tabName}-tab`).classList.add('active');
            });
        });
    }
    
    // History tab functionality
    function initHistoryTab() {
        const refreshHistoryBtn = document.getElementById('refresh-history-btn');
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener('click', loadDetectionHistory);
        }
        
        // Load history when tab is clicked
        const historyTabBtn = document.querySelector('.tab-btn[data-tab="history"]');
        if (historyTabBtn) {
            historyTabBtn.addEventListener('click', loadDetectionHistory);
        }
        
        // Initial load of history data
        loadDetectionHistory();
    }
    
    // Load detection history from database
    async function loadDetectionHistory() {
        try {
            const historyGallery = document.getElementById('history-gallery');
            if (!historyGallery) return;
            
            // Show loading indicator
            historyGallery.innerHTML = '<div class="loading-message">Loading history...</div>';
            
            const response = await fetch(`${apiBaseUrl}/history/detections`);
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('History data:', data);
            
            if (data.detections && Array.isArray(data.detections)) {
                historyResults = data.detections;
                displayHistoryResults();
            } else {
                historyGallery.innerHTML = '<div class="no-data-message">No detection history found</div>';
            }
        } catch (error) {
            console.error('Error loading detection history:', error);
            const historyGallery = document.getElementById('history-gallery');
            if (historyGallery) {
                historyGallery.innerHTML = `<div class="error-message">Error loading history: ${error.message}</div>`;
            }
            showNotification('Error loading detection history', 'error');
        }
    }
    
    // Display history results in UI
    function displayHistoryResults() {
        const historyGallery = document.getElementById('history-gallery');
        const historyCount = document.getElementById('history-count');
        
        if (!historyGallery) return;
        
        historyGallery.innerHTML = '';
        
        if (historyResults.length === 0) {
            historyGallery.innerHTML = '<div class="no-data-message">No detection history found</div>';
            if (historyCount) historyCount.textContent = '0';
            return;
        }
        
        historyResults.forEach(detection => {
            const resultItem = document.createElement('div');
            resultItem.className = 'history-item';
            
            // Create result image
            const resultImage = document.createElement('img');
            resultImage.src = detection.result_image_url || '/static/placeholder.svg';
            resultImage.alt = 'Detection Result';
            resultImage.className = 'history-image';
            resultImage.onerror = () => {
                resultImage.src = '/static/placeholder.svg';
                resultImage.alt = 'Image not available';
            };
            
            // Create result info
            const resultInfo = document.createElement('div');
            resultInfo.className = 'history-info';
            
            // Format the timestamp
            let timestamp = new Date();
            try {
                timestamp = new Date(detection.timestamp);
            } catch (e) {
                console.error('Error parsing timestamp:', e);
            }
            
            const formattedDate = timestamp.toLocaleDateString();
            const formattedTime = timestamp.toLocaleTimeString();
            
            // Format the confidence value
            const confidencePercentage = (detection.confidence * 100).toFixed(2);
            
            // Set result info content
            resultInfo.innerHTML = `
                <h4>${detection.file_name || 'Unknown file'}</h4>
                <p class="result-status ${detection.fall_detected ? 'fall-detected' : 'no-fall'}">
                    ${detection.fall_detected ? '⚠️ Fall Detected' : '✓ No Fall Detected'}
                </p>
                <p><strong>Date:</strong> ${formattedDate}</p>
                <p><strong>Time:</strong> ${formattedTime}</p>
                <p><strong>Model:</strong> ${detection.model_version || 'Unknown'}</p>
                <p><strong>Confidence:</strong> ${confidencePercentage}%</p>
                <p><strong>Processing Time:</strong> ${detection.processing_time ? detection.processing_time.toFixed(2) + ' ms' : 'N/A'}</p>
            `;
            
            // Create view buttons
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'history-buttons';
            
            // Original image button
            const originalButton = document.createElement('button');
            originalButton.className = 'history-button';
            originalButton.textContent = 'View Original';
            originalButton.addEventListener('click', () => {
                window.open(`${apiBaseUrl}/history/detection/${detection.id}/image/original`, '_blank');
            });
            
            // Processed image button
            const processedButton = document.createElement('button');
            processedButton.className = 'history-button';
            processedButton.textContent = 'View Processed';
            processedButton.addEventListener('click', () => {
                window.open(`${apiBaseUrl}/history/detection/${detection.id}/image/processed`, '_blank');
            });
            
            // Add buttons to container
            buttonContainer.appendChild(originalButton);
            buttonContainer.appendChild(processedButton);
            
            // Append elements to history item
            resultItem.appendChild(resultImage);
            resultItem.appendChild(resultInfo);
            resultItem.appendChild(buttonContainer);
            
            // Append history item to gallery
            historyGallery.appendChild(resultItem);
        });
        
        // Update history count
        if (historyCount) historyCount.textContent = historyResults.length;
    }
    
    // Process button functionality
    async function processFiles() {
        if (files.length === 0) {
            showNotification('No files to process', 'warning');
            return;
        }
        
        // Get selected model
        const modelSelect = document.getElementById('model-select');
        const selectedModel = modelSelect.value;
        
        // The model selection values are already in the server-expected format
        const serverModel = selectedModel; // Already in format 'YOLOv8' or 'YOLOv8-P2'
        const displayModel = selectedModel;
        
        // Clear previous results
        const resultsGallery = document.getElementById('results-gallery');
        resultsGallery.innerHTML = '<div class="loading-message">Processing files with ' + displayModel + '...</div>';
        
        // Reset benchmark results
        benchmarkResults = [];
        
        // Process each file
        for (const file of files) {
            try {
                await processFile(file, selectedModel);
            } catch (error) {
                console.error(`Error processing file ${file.name}:`, error);
                showNotification(`Error processing ${file.name}`, 'error');
            }
        }
        
        // Update benchmark charts
        updateCharts();
    }
    
    // Process a single file
    async function processFile(file, modelVersion) {
        const formData = new FormData();
        formData.append('file', file);
        
        // The model selection values are already in the server-expected format
        formData.append('model', modelVersion); // Already in format 'YOLOv8' or 'YOLOv8-P2'
        
        const startTime = performance.now();
        
        try {
            const response = await fetch(`${apiBaseUrl}/detect`, {
                method: 'POST',
                body: formData
            });
            
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('Detection result:', data);
            
            // Add to benchmark results
            benchmarkResults.push({
                fileName: file.name,
                fileType: file.type.startsWith('image/') ? 'Image' : 'Video',
                processingTime: processingTime,
                fallDetected: data.fall_detected,
                confidence: getConfidenceValue(data),
                modelVersion: modelVersion
            });
            
            // Display result
            displayResult(data, file.name, modelVersion);
            
            // Show notification
            const displayModel = data.model_version || modelVersion;
            showNotification(`Processed ${file.name} with ${displayModel} in ${processingTime.toFixed(2)}ms`, 'success');
            
            return data;
        } catch (error) {
            console.error('Error processing file:', error);
            showNotification(`Error: ${error.message}`, 'error');
            throw error;
        }
    }
    
    // Display result in UI
    function displayResult(data, fileName, modelVersion) {
        const resultsGallery = document.getElementById('results-gallery');
        
        // Remove loading message if present
        const loadingMessage = resultsGallery.querySelector('.loading-message');
        if (loadingMessage) {
            resultsGallery.removeChild(loadingMessage);
        }
        
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        const resultMediaContainer = document.createElement('div');
        resultMediaContainer.className = 'result-media-container';
        
        const resultMedia = document.createElement('img');
        resultMedia.src = data.result_image;
        resultMedia.alt = 'Result';
        resultMedia.className = 'result-media';
        
        resultMediaContainer.appendChild(resultMedia);
        
        const resultInfo = document.createElement('div');
        resultInfo.className = 'result-info';
        
        const confidence = getConfidenceValue(data);
        const confidencePercentage = (confidence * 100).toFixed(2);
        
        resultInfo.innerHTML = `
            <h4>${fileName}</h4>
            <p class="result-status ${data.fall_detected ? 'fall-detected' : 'no-fall'}">
                ${data.fall_detected ? '⚠️ Fall Detected' : '✓ No Fall Detected'}
            </p>
            <p><strong>Model:</strong> ${modelVersion}</p>
            <p><strong>Confidence:</strong> ${confidencePercentage}%</p>
        `;
        
        resultItem.appendChild(resultMediaContainer);
        resultItem.appendChild(resultInfo);
        
        resultsGallery.appendChild(resultItem);
    }
    
    // Create SVG upload icon
    function createUploadIcon() {
        const uploadIcon = document.getElementById('upload-icon');
        uploadIcon.outerHTML = `
            <svg id="upload-icon" width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 15V3M12 3L7 8M12 3L17 8" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M3 15V19C3 20.1046 3.89543 21 5 21H19C20.1046 21 21 20.1046 21 19V15" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        `;
    }
    
    // Initialize charts
    function initCharts() {
        const processingTimeCtx = document.getElementById('processing-time-chart')?.getContext('2d');
        const confidenceCtx = document.getElementById('confidence-chart')?.getContext('2d');
        
        if (!processingTimeCtx || !confidenceCtx) {
            console.error('Chart contexts not found');
            return;
        }
        
        // Processing time chart
        processingTimeChart = new Chart(processingTimeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Processing Time (ms)',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (ms)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Files'
                        }
                    }
                }
            }
        });
        
        // Confidence chart
        confidenceChart = new Chart(confidenceCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Confidence Score',
                    data: [],
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Confidence'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Files'
                        }
                    }
                }
            }
        });
    }
    
    // Add result to benchmark table
    function addToBenchmarkTable(result) {
        const tableBody = document.getElementById('benchmark-table-body');
        const row = document.createElement('tr');
        
        // Model version is already in display format (YOLOv8 or YOLOv8-P2)
        const displayModel = result.modelVersion;
        
        row.innerHTML = `
            <td>${result.fileName}</td>
            <td>${result.fileType}</td>
            <td>${displayModel}</td>
            <td>${result.processingTime.toFixed(2)}</td>
            <td>${result.fallDetected ? 'Yes' : 'No'}</td>
            <td>${(result.confidence * 100).toFixed(2)}%</td>
        `;
        
        tableBody.appendChild(row);
    }
    
    // Update charts with benchmark data
    function updateCharts() {
        // Check if charts are initialized
        if (!processingTimeChart || !confidenceChart) {
            console.error('Charts not initialized');
            return;
        }
        
        // Make sure we have data to display
        if (benchmarkResults.length === 0) {
            console.log('No benchmark results to display in charts');
            return;
        }
        
        console.log('Updating charts with data:', benchmarkResults);
        
        // Group results by model version
        const resultsByModel = {};
        benchmarkResults.forEach(result => {
            const modelVersion = result.modelVersion;
            if (!resultsByModel[modelVersion]) {
                resultsByModel[modelVersion] = [];
            }
            resultsByModel[modelVersion].push(result);
        });
        
        // Get unique file names across all results
        const allFileNames = [...new Set(benchmarkResults.map(result => result.fileName))];
        
        // Create datasets for processing time chart
        const processingTimeDatasets = [];
        const chartColors = {
            'YOLOv8': 'rgba(54, 162, 235, 0.6)',
            'YOLOv8-P2': 'rgba(255, 99, 132, 0.6)'
        };
        
        // Map for display names - not needed anymore as we're using the actual names
        const modelDisplayNames = {};
        
        // Create datasets for each model
        Object.keys(resultsByModel).forEach(modelVersion => {
            const modelResults = resultsByModel[modelVersion];
            const displayName = modelDisplayNames[modelVersion] || modelVersion;
            
            // Create processing time dataset
            const processingTimeData = allFileNames.map(fileName => {
                const result = modelResults.find(r => r.fileName === fileName);
                return result ? result.processingTime : null;
            });
            
            processingTimeDatasets.push({
                label: `${displayName} Processing Time (ms)`,
                data: processingTimeData,
                backgroundColor: modelColors[modelVersion]?.backgroundColor || 'rgba(128, 128, 128, 0.5)',
                borderColor: modelColors[modelVersion]?.borderColor || 'rgba(128, 128, 128, 1)',
                borderWidth: 1
            });
        });
        
        // Update processing time chart
        processingTimeChart.data.labels = allFileNames;
        processingTimeChart.data.datasets = processingTimeDatasets;
        processingTimeChart.update();
        
        // Create datasets for confidence chart
        const confidenceDatasets = [];
        
        Object.keys(resultsByModel).forEach(modelVersion => {
            const modelResults = resultsByModel[modelVersion];
            const displayName = modelDisplayNames[modelVersion] || modelVersion;
            
            // Create confidence dataset
            const confidenceData = allFileNames.map(fileName => {
                const result = modelResults.find(r => r.fileName === fileName);
                return result ? result.confidence * 100 : null;
            });
            
            confidenceDatasets.push({
                label: `${displayName} Confidence (%)`,
                data: confidenceData,
                backgroundColor: modelColors[modelVersion]?.backgroundColor || 'rgba(128, 128, 128, 0.5)',
                borderColor: modelColors[modelVersion]?.borderColor || 'rgba(128, 128, 128, 1)',
                borderWidth: 1
            });
        });
        
        // Update confidence chart
        confidenceChart.data.labels = allFileNames;
        confidenceChart.data.datasets = confidenceDatasets;
        confidenceChart.update();
        
        // Create model comparison summary
        createModelComparisonSummary(resultsByModel, modelDisplayNames);
        
        // Switch to benchmark tab to show results
        document.querySelector('.tab-btn[data-tab="benchmark"]').click();
        
        // Update benchmark table
        updateBenchmarkTable();
    }
    
    // Update benchmark table with all results
    function updateBenchmarkTable() {
        const tableBody = document.getElementById('benchmark-table-body');
        if (!tableBody) return;
        
        // Clear existing table rows
        tableBody.innerHTML = '';
        
        // Add each result to the table
        benchmarkResults.forEach(result => {
            addToBenchmarkTable(result);
        });
    }
    
    // Create model comparison summary
    function createModelComparisonSummary(resultsByModel, modelDisplayNames) {
        const benchmarkTab = document.getElementById('benchmark-tab');
        if (!benchmarkTab) return;
        
        // Remove existing summary if any
        const existingSummary = document.getElementById('model-comparison-summary');
        if (existingSummary) {
            existingSummary.remove();
        }
        
        // Create summary container
        const summaryContainer = document.createElement('div');
        summaryContainer.id = 'model-comparison-summary';
        summaryContainer.className = 'model-comparison-summary';
        
        // Create heading
        const heading = document.createElement('h3');
        heading.textContent = 'Model Comparison Summary';
        summaryContainer.appendChild(heading);
        
        // Calculate average metrics for each model
        const modelMetrics = {};
        
        Object.keys(resultsByModel).forEach(modelVersion => {
            const results = resultsByModel[modelVersion];
            const displayName = modelDisplayNames[modelVersion] || modelVersion;
            
            const avgProcessingTime = results.reduce((sum, result) => sum + result.processingTime, 0) / results.length;
            const avgConfidence = results.reduce((sum, result) => sum + result.confidence, 0) / results.length * 100;
            const fallDetectionRate = results.filter(result => result.fallDetected).length / results.length * 100;
            
            modelMetrics[modelVersion] = {
                displayName,
                avgProcessingTime,
                avgConfidence,
                fallDetectionRate,
                count: results.length
            };
        });
        
        // Create comparison table
        const table = document.createElement('table');
        table.className = 'model-comparison-table';
        
        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        ['Model', 'Files Processed', 'Avg. Processing Time (ms)', 'Avg. Confidence (%)', 'Fall Detection Rate (%)'].forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        Object.keys(modelMetrics).forEach(modelVersion => {
            const metrics = modelMetrics[modelVersion];
            const row = document.createElement('tr');
            
            // Model name
            const modelCell = document.createElement('td');
            modelCell.textContent = metrics.displayName;
            row.appendChild(modelCell);
            
            // Files processed
            const countCell = document.createElement('td');
            countCell.textContent = metrics.count;
            row.appendChild(countCell);
            
            // Avg processing time
            const timeCell = document.createElement('td');
            timeCell.textContent = metrics.avgProcessingTime.toFixed(2);
            row.appendChild(timeCell);
            
            // Avg confidence
            const confidenceCell = document.createElement('td');
            confidenceCell.textContent = metrics.avgConfidence.toFixed(2);
            row.appendChild(confidenceCell);
            
            // Fall detection rate
            const rateCell = document.createElement('td');
            rateCell.textContent = metrics.fallDetectionRate.toFixed(2);
            row.appendChild(rateCell);
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        summaryContainer.appendChild(table);
        
        // Add summary to benchmark tab before the benchmark table container
        const benchmarkTableContainer = document.querySelector('.benchmark-table-container');
        if (benchmarkTableContainer) {
            benchmarkTab.insertBefore(summaryContainer, benchmarkTableContainer);
        } else {
            benchmarkTab.appendChild(summaryContainer);
        }
    }
    
    // Helper function to extract confidence value from API response
    function getConfidenceValue(data) {
        // Try to get confidence directly
        if (data.confidence !== undefined && data.confidence !== null) {
            const conf = parseFloat(data.confidence);
            if (!isNaN(conf)) {
                return conf;
            }
        }
        
        // Try to get confidence from fall events
        if (data.fall_events && data.fall_events.length > 0) {
            const confidences = data.fall_events
                .map(e => {
                    if (e.confidence !== undefined && e.confidence !== null) {
                        return parseFloat(e.confidence);
                    }
                    return 0;
                })
                .filter(c => !isNaN(c));
                
            if (confidences.length > 0) {
                return Math.max(...confidences);
            }
        }
        
        // Default fallback value
        return data.fall_detected || (data.fall_events && data.fall_events.length > 0) ? 0.75 : 0;
    }
    
    // Show notification
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
});
