document.addEventListener('DOMContentLoaded', () => {
    // --- Chat Elements ---
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const fileUpload = document.getElementById('file-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // --- Docs Elements ---
    const docsList = document.getElementById('docs-list');
    const docsUploadBtn = document.getElementById('docs-upload-btn');
    const docsFileInput = document.getElementById('docs-file-input');

    // --- Settings Elements ---
    const settingUserId = document.getElementById('setting-user-id');
    const settingThreadId = document.getElementById('setting-thread-id');
    const resetSessionBtn = document.getElementById('reset-session-btn');

    // --- Navigation Layout ---
    const sidebar = document.querySelector('.sidebar');
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');

    // --- State ---
    let currentFile = null;
    let threadId = localStorage.getItem('coursegpt_thread_id');
    let userId = localStorage.getItem('coursegpt_user_id');

    // Initialize Session
    if (!threadId || !userId) {
        threadId = 'thread_' + Date.now();
        userId = 'user_' + Math.floor(Math.random() * 1000);
        localStorage.setItem('coursegpt_thread_id', threadId);
        localStorage.setItem('coursegpt_user_id', userId);
    }

    // Update Settings View
    if (settingUserId) settingUserId.value = userId;
    if (settingThreadId) settingThreadId.value = threadId;

    // --- Navigation ---
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetViewId = item.getAttribute('data-view');

            // Update Nav
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Update View
            views.forEach(view => {
                if (view.id === targetViewId) {
                    view.classList.remove('hidden');
                    view.classList.add('active');
                } else {
                    view.classList.add('hidden');
                    view.classList.remove('active');
                }
            });

            // Load data if needed
            if (targetViewId === 'docs-view') {
                loadDocuments();
            }

            // Collapse mobile sidebar after navigation
            if (sidebar && sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        });
    });

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', () => {
            if (!sidebar) return;
            sidebar.classList.toggle('open');
        });

        document.addEventListener('click', (event) => {
            if (!sidebar) return;
            const clickInsideSidebar = sidebar.contains(event.target);
            const clickOnToggle = mobileMenuToggle.contains(event.target);
            if (!clickInsideSidebar && !clickOnToggle) {
                sidebar.classList.remove('open');
            }
        });
    }

    // --- Chat Logic ---

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        sendBtn.disabled = this.value.trim() === '';
    });

    // Handle Enter key
    userInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim() !== '') {
                sendMessage();
            }
        }
    });

    // File Upload Handling (Chat)
    uploadBtn.addEventListener('click', () => fileUpload.click());

    fileUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            currentFile = e.target.files[0];
            fileName.textContent = currentFile.name;
            filePreview.classList.remove('hidden');
        }
    });

    removeFileBtn.addEventListener('click', () => {
        currentFile = null;
        fileUpload.value = '';
        filePreview.classList.add('hidden');
    });

    // Send Message
    sendBtn.addEventListener('click', sendMessage);

    // Clear Chat
    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message ai-message">
                <div class="message-content">
                    <p>👋 Hello! I'm CourseGPT. I can help you with your course materials, solve math problems, or analyze documents. How can I assist you today?</p>
                </div>
            </div>
        `;
    });

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text && !currentFile) return;

        // Add User Message
        addMessage(text, 'user');

        // Reset Input
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Create placeholder for AI message
        const aiMessageId = 'ai-msg-' + Date.now();
        createAiMessagePlaceholder(aiMessageId);

        try {
            const formData = new FormData();
            formData.append('prompt', text);
            formData.append('thread_id', threadId);
            formData.append('user_id', userId);
            if (currentFile) {
                formData.append('file', currentFile);
                currentFile = null;
                fileUpload.value = '';
                filePreview.classList.add('hidden');
            }

            const response = await fetch('/graph/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') break;

                        try {
                            const data = JSON.parse(dataStr);

                            if (data.type === 'handoff') {
                                renderHandoff(aiMessageId, data.content);
                            } else if (data.type === 'token') {
                                aiText += data.content; // In this case, content is the full message chunk from the node
                                updateAiMessage(aiMessageId, aiText);
                            } else if (data.type === 'error') {
                                console.error('Stream error:', data.content);
                                updateAiMessage(aiMessageId, aiText + '\n\n*Error: ' + data.content + '*');
                            }
                        } catch (e) {
                            console.error('Error parsing SSE:', e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Error:', error);
            updateAiMessage(aiMessageId, 'Sorry, something went wrong. Please try again.');
        }
    }

    function createAiMessagePlaceholder(id) {
        const messageDiv = document.createElement('div');
        messageDiv.id = id;
        messageDiv.classList.add('message', 'ai-message');
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span>.</span><span>.</span><span>.</span>
                </div>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function renderHandoff(messageId, content) {
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) return;

        const handoffTarget = content.handoff || 'Specialized Agent';
        const taskSummary = content.task_summary || '';
        const rationale = content.route_rationale || '';

        const handoffHtml = `
            <div class="handoff-card">
                <div class="handoff-header">
                    <i data-feather="shuffle"></i>
                    <span>Handoff to ${handoffTarget.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                </div>
                <div class="handoff-summary">${taskSummary}</div>
                <div class="handoff-rationale">Rationale: ${rationale}</div>
            </div>
        `;

        // Insert before the message content
        const contentDiv = messageDiv.querySelector('.message-content');
        contentDiv.insertAdjacentHTML('beforebegin', handoffHtml);
        feather.replace();
    }

    function updateAiMessage(id, text) {
        const messageDiv = document.getElementById(id);
        if (!messageDiv) return;

        const contentDiv = messageDiv.querySelector('.message-content');

        // Parse Markdown and LaTeX
        contentDiv.innerHTML = marked.parse(text);

        // Highlight code
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        // Render LaTeX
        renderMathInElement(contentDiv, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true }
            ],
            throwOnError: false
        });

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Deprecated: addMessage (kept for simple user messages)
    function addMessage(text, sender) {
        if (sender === 'ai') return; // AI messages handled by streaming

        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.innerHTML = `<div class="message-content">${marked.parse(text)}</div>`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- Documents Logic ---

    docsUploadBtn.addEventListener('click', () => docsFileInput.click());

    docsFileInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const formData = new FormData();
            formData.append('file', file);

            try {
                // Show loading state
                docsUploadBtn.disabled = true;
                docsUploadBtn.innerHTML = '<i data-feather="loader" class="spin"></i> Uploading...';
                feather.replace();

                const response = await fetch('/files/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Upload failed');

                // Refresh list
                loadDocuments();

            } catch (error) {
                console.error('Upload error:', error);
                alert('Failed to upload document');
            } finally {
                docsUploadBtn.disabled = false;
                docsUploadBtn.innerHTML = '<i data-feather="upload"></i> Upload Document';
                feather.replace();
                docsFileInput.value = '';
            }
        }
    });

    async function loadDocuments() {
        try {
            const response = await fetch('/files/');
            const data = await response.json();
            renderDocuments(data.files);
        } catch (error) {
            console.error('Load docs error:', error);
            docsList.innerHTML = '<div class="empty-state"><p>Failed to load documents</p></div>';
        }
    }

    function renderDocuments(files) {
        if (!files || files.length === 0) {
            docsList.innerHTML = `
                <div class="empty-state">
                    <i data-feather="file-text"></i>
                    <p>No documents yet. Upload one to get started.</p>
                </div>
            `;
            feather.replace();
            return;
        }

        docsList.innerHTML = files.map(file => `
            <div class="doc-card">
                <div class="doc-icon">
                    <i data-feather="file-text"></i>
                </div>
                <div class="doc-info">
                    <h3>${file.key}</h3>
                    <p>${formatSize(file.size)} • ${new Date(file.last_modified).toLocaleDateString()}</p>
                </div>
                <div class="doc-actions">
                    <button class="icon-btn" onclick="viewDocument('${file.key}')" title="View">
                        <i data-feather="eye"></i>
                    </button>
                    <button class="icon-btn" onclick="deleteDocument('${file.key}')" title="Delete">
                        <i data-feather="trash-2"></i>
                    </button>
                </div>
            </div>
        `).join('');

        feather.replace();
    }

    window.viewDocument = async (key) => {
        try {
            const response = await fetch(`/files/view/${encodeURIComponent(key)}`);
            if (!response.ok) throw new Error('Unable to generate view link');

            const data = await response.json();
            if (data.url) {
                window.open(data.url, '_blank', 'noopener');
            }
        } catch (error) {
            console.error('View error:', error);
            alert('Unable to open document preview');
        }
    };

    window.deleteDocument = async (key) => {
        if (!confirm(`Are you sure you want to delete "${key}"?`)) return;

        try {
            const response = await fetch(`/files/${encodeURIComponent(key)}`, {
                method: 'DELETE'
            });

            if (!response.ok) throw new Error('Delete failed');
            loadDocuments();
        } catch (error) {
            console.error('Delete error:', error);
            alert('Failed to delete document');
        }
    };

    function formatSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    // --- Settings Logic ---

    resetSessionBtn.addEventListener('click', () => {
        if (confirm('This will clear your chat history and generate a new session ID. Continue?')) {
            localStorage.removeItem('coursegpt_thread_id');
            localStorage.removeItem('coursegpt_user_id');
            location.reload();
        }
    });
});
