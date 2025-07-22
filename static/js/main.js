document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const elements = {
        nameInput: document.getElementById('name-input'),
        registerBtn: document.getElementById('register-btn'),
        retrainBtn: document.getElementById('retrain-btn'),
        refreshCameraBtn: document.getElementById('refresh-camera'),
        userListContainer: document.getElementById('user-list-container'),
        eventLog: document.getElementById('event-log'),
        clearLogBtn: document.getElementById('clear-log'),
        cancelCaptureBtn: document.getElementById('cancel-capture'),
        
        cameraStatusText: document.getElementById('camera-status-text'),
        cameraDetails: document.getElementById('camera-details'),
        cameraBackend: document.getElementById('camera-backend'),
        cameraIndex: document.getElementById('camera-index'),
        cameraFeedback: document.getElementById('camera-feedback'),
        
        overlay: document.getElementById('capture-overlay'),
        overlayTitle: document.getElementById('overlay-title'),
        overlayText: document.getElementById('overlay-text'),
        progressBarInner: document.getElementById('progress-bar-inner'),
        
        videoFeed: document.getElementById('video-feed'),
        canvas: document.getElementById('capture-canvas'),
        
        userCount: document.getElementById('user-count'),
        modelStatus: document.getElementById('model-status'),
        cameraStatus: document.getElementById('camera-status'),
        statusLight: document.getElementById('status-light'),
        statusText: document.getElementById('status-text'),
        
        registerFeedback: document.getElementById('register-feedback'),
        retrainFeedback: document.getElementById('retrain-feedback')
    };

    // Constants
    const constants = {
        SAMPLES_TO_CAPTURE: 5,
        CAPTURE_INTERVAL: 500, // ms
        STATUS_UPDATE_INTERVAL: 3000, // ms
        MAX_RECONNECT_ATTEMPTS: 3
    };

    // State
    const state = {
        captureInterval: null,
        systemStatus: {
            camera_initialized: false,
            model_trained: false,
            registered_users: 0
        },
        captureCancelled: false
    };

    // Functions
    function logEvent(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;
        
        elements.eventLog.appendChild(logEntry);
        elements.eventLog.scrollTop = elements.eventLog.scrollHeight;
    }

    async function updateSystemStatus() {
        try {
            const response = await fetch('/system_status');
            const data = await response.json();
            
            state.systemStatus = data;
            updateUI(data);
            
        } catch (error) {
            logEvent(`Failed to update system status: ${error}`, 'error');
        }
    }

    function updateUI(data) {
        // Update user count and model status
        elements.userCount.textContent = data.registered_users || 0;
        elements.modelStatus.textContent = data.model_trained ? 'Yes' : 'No';
        
        // Update camera status
        if (data.camera_initialized) {
            elements.cameraStatus.textContent = 'Online';
            elements.cameraStatus.style.color = 'var(--secondary)';
            elements.cameraStatusText.innerHTML = `<span style="color: var(--secondary)">✓ Camera Connected</span>`;
            elements.cameraDetails.classList.remove('hidden');
            elements.cameraBackend.textContent = data.camera_backend || 'Default';
            elements.cameraIndex.textContent = data.camera_index !== undefined ? data.camera_index : 'Auto';
        } else {
            elements.cameraStatus.textContent = 'Offline';
            elements.cameraStatus.style.color = 'var(--danger)';
            elements.cameraStatusText.innerHTML = `<span style="color: var(--danger)">✗ Camera Error</span>`;
            elements.cameraDetails.classList.add('hidden');
            
            if (data.last_error) {
                const errorLines = data.last_error.split('\n');
                elements.cameraStatusText.innerHTML += `<br><small>${errorLines[0]}</small>`;
            }
        }
        
        // Update system status indicator
        if (data.last_error) {
            elements.statusLight.className = 'status-indicator error';
            elements.statusText.textContent = 'System Error';
            elements.statusText.style.color = 'var(--danger)';
        } else if (!data.camera_initialized || !data.model_trained) {
            elements.statusLight.className = 'status-indicator warning';
            elements.statusText.textContent = 'System Warning';
            elements.statusText.style.color = 'var(--warning)';
        } else {
            elements.statusLight.className = 'status-indicator';
            elements.statusText.textContent = 'System Active';
            elements.statusText.style.color = 'var(--secondary)';
        }
    }

    async function fetchUsers() {
        try {
            const response = await fetch('/get_users');
            const data = await response.json();
            
            if (data.status === 'success') {
                renderUserList(data.users);
                logEvent('User list updated');
            } else {
                logEvent(data.message || 'Failed to fetch users', 'error');
            }
        } catch (error) {
            logEvent(`Failed to fetch users: ${error}`, 'error');
        }
    }

    function renderUserList(users) {
        elements.userListContainer.innerHTML = '';
        
        if (!users || users.length === 0) {
            elements.userListContainer.innerHTML = '<p>No users registered</p>';
            return;
        }
        
        const userList = document.createElement('ul');
        userList.className = 'user-list';
        
        users.forEach(user => {
            const userItem = document.createElement('li');
            userItem.innerHTML = `
                <span>${user}</span>
                <button class="delete-btn" data-name="${user}" title="Delete user">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            `;
            userList.appendChild(userItem);
        });
        
        elements.userListContainer.appendChild(userList);
    }

    function startMultiSampleCapture(name) {
        if (!elements.videoFeed.complete || elements.videoFeed.naturalWidth === 0) {
            logEvent('Video feed not loaded or camera error', 'error');
            elements.registerFeedback.textContent = 'Camera not available';
            elements.registerFeedback.style.color = 'var(--danger)';
            return;
        }
        
        elements.registerBtn.disabled = true;
        elements.registerFeedback.textContent = 'Registration in progress...';
        elements.registerFeedback.style.color = 'var(--text-secondary)';
        
        logEvent(`Starting registration for ${name}`);
        elements.overlay.classList.remove('hidden');
        state.captureCancelled = false;
        
        const capturedImages = [];
        let samplesTaken = 0;

        elements.cancelCaptureBtn.onclick = () => {
            state.captureCancelled = true;
            logEvent('Registration cancelled by user', 'warning');
            elements.registerFeedback.textContent = 'Registration cancelled';
            elements.registerFeedback.style.color = 'var(--warning)';
            elements.overlay.classList.add('hidden');
            elements.registerBtn.disabled = false;
        };

        const captureInterval = setInterval(() => {
            if (state.captureCancelled || samplesTaken >= constants.SAMPLES_TO_CAPTURE) {
                clearInterval(captureInterval);
                
                if (!state.captureCancelled && capturedImages.length > 0) {
                    sendImagesToServer(name, capturedImages);
                }
                return;
            }
            
            samplesTaken++;
            const progress = (samplesTaken / constants.SAMPLES_TO_CAPTURE) * 100;
            
            elements.overlayTitle.textContent = `Capturing... ${samplesTaken}/${constants.SAMPLES_TO_CAPTURE}`;
            elements.overlayText.textContent = "Move your head slightly for better results";
            elements.progressBarInner.style.width = `${progress}%`;

            // Capture image
            const ctx = elements.canvas.getContext('2d');
            elements.canvas.width = elements.videoFeed.naturalWidth;
            elements.canvas.height = elements.videoFeed.naturalHeight;
            ctx.drawImage(elements.videoFeed, 0, 0, elements.canvas.width, elements.canvas.height);
            capturedImages.push(elements.canvas.toDataURL('image/jpeg'));
            
            logEvent(`Captured sample ${samplesTaken} for ${name}`);

        }, constants.CAPTURE_INTERVAL);
    }
    
    async function sendImagesToServer(name, images) {
        elements.overlayTitle.textContent = 'Processing & Training...';
        elements.overlayText.textContent = 'This may take a moment';
        logEvent('Sending samples to server for training...');
        
        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, images }),
            });
            
            const result = await response.json();
            
            logEvent(result.message, result.status);
            elements.registerFeedback.textContent = result.message;
            
            if (result.status === 'success') {
                elements.registerFeedback.style.color = 'var(--secondary)';
                elements.nameInput.value = '';
                await fetchUsers();
            } else {
                elements.registerFeedback.style.color = 'var(--danger)';
            }
        } catch(error) {
            logEvent(`Registration failed: ${error}`, 'error');
            elements.registerFeedback.textContent = 'Registration failed';
            elements.registerFeedback.style.color = 'var(--danger)';
        } finally {
            elements.overlay.classList.add('hidden');
            elements.registerBtn.disabled = false;
            updateSystemStatus();
        }
    }

    async function retrainModel() {
        elements.retrainBtn.disabled = true;
        elements.retrainFeedback.textContent = 'Retraining model...';
        elements.retrainFeedback.style.color = 'var(--text-secondary)';
        
        logEvent('Starting model retraining...');
        
        try {
            const response = await fetch('/retrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            
            const result = await response.json();
            
            logEvent(result.message, result.status);
            elements.retrainFeedback.textContent = result.message;
            
            if (result.status === 'success') {
                elements.retrainFeedback.style.color = 'var(--secondary)';
            } else {
                elements.retrainFeedback.style.color = 'var(--danger)';
            }
        } catch(error) {
            logEvent(`Retraining failed: ${error}`, 'error');
            elements.retrainFeedback.textContent = 'Retraining failed';
            elements.retrainFeedback.style.color = 'var(--danger)';
        } finally {
            elements.retrainBtn.disabled = false;
            updateSystemStatus();
        }
    }

    async function reconnectCamera() {
        elements.refreshCameraBtn.disabled = true;
        elements.refreshCameraBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Connecting...';
        elements.cameraFeedback.textContent = 'Attempting to reconnect...';
        elements.cameraFeedback.style.color = 'var(--text-secondary)';
        
        try {
            const response = await fetch('/camera_reconnect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            
            const result = await response.json();
            
            logEvent(result.message, result.status);
            elements.cameraFeedback.textContent = result.message;
            
            if (result.status === 'success') {
                elements.cameraFeedback.style.color = 'var(--secondary)';
                // Refresh video feed
                elements.videoFeed.src = elements.videoFeed.src;
            } else {
                elements.cameraFeedback.style.color = 'var(--danger)';
            }
        } catch(error) {
            logEvent(`Reconnection failed: ${error}`, 'error');
            elements.cameraFeedback.textContent = 'Reconnection failed';
            elements.cameraFeedback.style.color = 'var(--danger)';
        } finally {
            elements.refreshCameraBtn.disabled = false;
            elements.refreshCameraBtn.innerHTML = '<i class="fa-solid fa-rotate"></i> Reconnect Camera';
            updateSystemStatus();
        }
    }

    // Event Listeners
    elements.userListContainer.addEventListener('click', async (e) => {
        const deleteBtn = e.target.closest('.delete-btn');
        if (deleteBtn) {
            const name = deleteBtn.dataset.name;
            if (confirm(`Delete ${name}? This will retrain the model.`)) {
                logEvent(`Deleting user: ${name}...`);
                
                try {
                    const response = await fetch('/delete_user', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name }),
                    });
                    
                    const result = await response.json();
                    logEvent(result.message, result.status);
                    
                    if (result.status === 'success') {
                        await fetchUsers();
                    }
                } catch (error) {
                    logEvent(`Error deleting user: ${error}`, 'error');
                }
            }
        }
    });

    elements.registerBtn.addEventListener('click', () => {
        const name = elements.nameInput.value.trim();
        
        if (!name) {
            logEvent('Please enter a name', 'error');
            elements.registerFeedback.textContent = 'Name is required';
            elements.registerFeedback.style.color = 'var(--danger)';
            return;
        }
        
        if (name.length > 50) {
            logEvent('Name too long (max 50 chars)', 'error');
            elements.registerFeedback.textContent = 'Name too long';
            elements.registerFeedback.style.color = 'var(--danger)';
            return;
        }
        
        startMultiSampleCapture(name);
    });

    elements.retrainBtn.addEventListener('click', retrainModel);
    elements.refreshCameraBtn.addEventListener('click', reconnectCamera);
    
    elements.clearLogBtn.addEventListener('click', () => {
        elements.eventLog.innerHTML = '';
        logEvent('Log cleared');
    });

    // Initialize
    function initialize() {
        logEvent('System initializing...');
        
        // Set up periodic status updates
        updateSystemStatus();
        setInterval(updateSystemStatus, constants.STATUS_UPDATE_INTERVAL);
        
        // Load initial data
        fetchUsers();
        
        // Handle video feed errors
        elements.videoFeed.onerror = () => {
            logEvent('Video feed error', 'error');
        };
        
        logEvent('System ready');
    }
    
    initialize();
});