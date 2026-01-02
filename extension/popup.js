document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const statusBox = document.getElementById('statusBox');
    const resultArea = document.getElementById('resultArea');
    const errorMsg = document.getElementById('errorMsg');
    
    resultArea.classList.add('hidden');
    errorMsg.classList.add('hidden');
    statusBox.textContent = "Fetching text...";

    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: getSelectionText,
    }, async (results) => {
        const selectedText = results[0].result;

        if (!selectedText || selectedText.trim().length < 10) {
            statusBox.textContent = "Please select a longer text fragment (min. 10 characters).";
            return;
        }

        statusBox.textContent = "Analyzing...";

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: selectedText })
            });

            if (!response.ok) {
                throw new Error("Server error: " + response.status);
            }

            const data = await response.json();
            
            displayResult(data);
            statusBox.textContent = "Analysis complete.";

        } catch (error) {
            console.error(error);
            errorMsg.textContent = "Could not connect to the API. Did you start 'api.py'?";
            errorMsg.classList.remove('hidden');
            statusBox.textContent = "Error.";
        }
    });
});

function getSelectionText() {
    return window.getSelection().toString();
}

function displayResult(data) {
    const resultArea = document.getElementById('resultArea');
    const labelDiv = document.getElementById('predictionLabel');
    const confidenceSpan = document.getElementById('confidenceScore');
    const modelSpan = document.getElementById('modelName');
    const progressBar = document.getElementById('confidenceBar'); 

    labelDiv.textContent = data.label;
    
    labelDiv.className = 'label'; 
    progressBar.className = 'progress-fill';

    if (data.label === 'REAL') {
        labelDiv.classList.add('real');
        progressBar.style.backgroundColor = '#27AE60';
    } else {
        labelDiv.classList.add('fake');
        progressBar.style.backgroundColor = '#D32F2F';
    }

    const percentage = (data.score * 100).toFixed(1) + '%';
    confidenceSpan.textContent = percentage;
    
    progressBar.style.width = percentage; 

    const routerInfo = data.router_info ? ` | ${data.router_info}` : '';
    modelSpan.textContent = `${data.model}${routerInfo}`;

    resultArea.classList.remove('hidden');
}