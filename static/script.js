document.getElementById('analysis-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const analyzeBtn = document.getElementById('analyze-btn');
    const originalText = analyzeBtn.textContent;
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;

    const data = {
        molwt: document.getElementById('molwt').value,
        logp: document.getElementById('logp').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            document.getElementById('result-area').classList.remove('hidden');
            document.getElementById('lc50-value').textContent = result.prediction;

            const statusEl = document.getElementById('status-value');
            statusEl.textContent = result.status;

            if (result.status.includes('TOXIC') && !result.status.includes('Non-Toxic')) {
                statusEl.classList.add('toxic');
                statusEl.style.color = '#ff6b6b';
            } else {
                statusEl.classList.remove('toxic');
                statusEl.style.color = '#50c878';
            }
        } else {
            alert('Error: ' + result.error);
        }

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while connecting to the server.');
    } finally {
        analyzeBtn.textContent = originalText;
        analyzeBtn.disabled = false;
    }
});
