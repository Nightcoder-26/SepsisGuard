/**
 * SepsisGuard v4.0 - Clinical OS Logic
 */

const API_BASE = "http://localhost:5000";

// DOM Elements
const vitalsForm = document.getElementById("vitals-form");
const riskScoreEl = document.getElementById("risk-score");
const riskLabelEl = document.getElementById("risk-label");
const gaugeFillEl = document.getElementById("gauge-fill");
const alertBox = document.getElementById("alert-box");
const vitalsSummaryList = document.getElementById("vitals-summary");
const simulateBtn = document.getElementById("simulate-btn");
const apiStatusEl = document.getElementById("api-status");
const modelAccEl = document.getElementById("model-acc");

// Chart.js instance
let trendChart;
const trendData = {
    labels: Array(10).fill(''),
    datasets: [{
        label: 'Risk Progression (%)',
        data: Array(10).fill(0),
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56, 189, 248, 0.05)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: '#38bdf8'
    }]
};

// Initialize
document.addEventListener("DOMContentLoaded", async () => {
    initChart();
    if (document.getElementById('pieChart')) initPieChart();
    checkHealth();
    
    // Live Real-Time Evaluation Binding
    let liveDebounce;
    const allInputs = document.querySelectorAll("#vitals-form input");
    allInputs.forEach(input => {
        input.addEventListener("input", (e) => {
            if (e.target.id === "marker") {
                document.getElementById("marker-val").textContent = parseFloat(e.target.value).toFixed(2);
            }
            
            clearTimeout(liveDebounce);
            liveDebounce = setTimeout(() => {
                const data = {
                    Heart_Rate: parseFloat(document.getElementById("hr").value || 0),
                    Temperature: parseFloat(document.getElementById("temp").value || 0),
                    Blood_Pressure: parseFloat(document.getElementById("bp").value || 0),
                    Resp_Rate: parseFloat(document.getElementById("rr").value || 0),
                    Oxygen_Level: parseFloat(document.getElementById("spo2").value || 0),
                    Age: parseInt(document.getElementById("age").value || 0),
                    Infection_Marker: parseFloat(document.getElementById("marker").value || 0)
                };
                getPrediction(data);
            }, 300); // 300ms real-time latency
        });
    });
});

// Form Submission
vitalsForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = {
        Heart_Rate: parseFloat(document.getElementById("hr").value),
        Temperature: parseFloat(document.getElementById("temp").value),
        Blood_Pressure: parseFloat(document.getElementById("bp").value),
        Resp_Rate: parseFloat(document.getElementById("rr").value),
        Oxygen_Level: parseFloat(document.getElementById("spo2").value),
        Age: parseInt(document.getElementById("age").value),
        Infection_Marker: parseFloat(document.getElementById("marker").value)
    };
    
    await getPrediction(data);
});

// Simulation logic
simulateBtn.addEventListener("click", () => {
    document.getElementById("hr").value = 118;
    document.getElementById("temp").value = 39.5;
    document.getElementById("bp").value = 82;
    document.getElementById("rr").value = 28;
    document.getElementById("spo2").value = 88;
    document.getElementById("marker").value = 0.9;
    
    vitalsForm.dispatchEvent(new Event('submit'));
});

// Health check
async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        if (resp.ok) {
            apiStatusEl.textContent = "Neural Node Online";
            apiStatusEl.parentElement.querySelector('.dot').style.background = "#10b981";
        }
    } catch {
        apiStatusEl.textContent = "Node Offline";
        apiStatusEl.parentElement.querySelector('.dot').style.background = "#ef4444";
    }
}

// Prediction Logic
async function getPrediction(vitals) {
    const btn = document.getElementById("predict-btn");
    btn.classList.add("btn-loading");
    
    try {
        const resp = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(vitals)
        });
        
        const result = await resp.json();
        if (resp.ok) {
            updateUI(result);
        }
    } catch (err) {
        console.error(err);
    } finally {
        btn.classList.remove("btn-loading");
    }
}

function updateUI(res) {
    const score = res.risk_score;
    
    // 1. Circular Gauge Animation (Full circle = 251.2)
    const circumference = 251.2;
    const offset = ((100 - score) / 100) * circumference;
    gaugeFillEl.style.strokeDasharray = `${circumference} ${circumference}`;
    gaugeFillEl.style.strokeDashoffset = offset;
    
    // Color & Class handling
    let accentColor = "#10b981"; // Success
    if (score > 70) {
        accentColor = "#ef4444"; // Danger
        gaugeFillEl.classList.add("critical-glow");
        alertBox.classList.remove("hidden");
    } else if (score > 30) {
        accentColor = "#f59e0b"; // Warning
        gaugeFillEl.classList.remove("critical-glow");
        alertBox.classList.add("hidden");
    } else {
        gaugeFillEl.classList.remove("critical-glow");
        alertBox.classList.add("hidden");
    }
    
    gaugeFillEl.style.stroke = accentColor;
    
    // Score & Label
    animateCounter(riskScoreEl, score);
    riskLabelEl.textContent = res.risk_level;
    riskLabelEl.style.color = accentColor;
    
    // Calculate Advanced Metrics
    const hr = parseFloat(document.getElementById("hr").value);
    const temp = parseFloat(document.getElementById("temp").value);
    const rr = parseFloat(document.getElementById("rr").value);
    const marker = parseFloat(document.getElementById("marker").value);
    
    let sirsCount = 0;
    if (temp < 36 || temp > 38) sirsCount++;
    if (hr > 90) sirsCount++;
    if (rr > 20) sirsCount++;
    if (marker > 0.5) sirsCount++; 
    document.getElementById("sirs-score").textContent = `${sirsCount}/4`;
    
    // Survival Rate Estimate
    let survival = 99.8 - (score * 0.45); 
    if (survival < 15) survival = 15; 
    const survivalEl = document.getElementById("survival-rate");
    if (survivalEl) survivalEl.textContent = survival.toFixed(1) + "%";

    let confidence = 50 + (Math.abs(score - 50) / 50) * 48.4; 
    if (confidence > 98.4) confidence = 98.4;
    document.getElementById("confidence-score").textContent = confidence.toFixed(1) + "%";

    let onset = "> 12 hrs";
    if (score > 80) onset = "1 - 3 hrs";
    else if (score > 60) onset = "4 - 6 hrs";
    else if (score > 30) onset = "8 - 12 hrs";
    document.getElementById("onset-time").textContent = onset;

    // Update Ranges UI
    const bp = parseFloat(document.getElementById("bp").value);
    const spo2 = parseFloat(document.getElementById("spo2").value);
    
    if (document.getElementById("range-hr")) {
        document.getElementById("range-hr").textContent = hr;
        document.getElementById("fill-hr").style.width = `${Math.min((hr / 200) * 100, 100)}%`;
        document.getElementById("fill-hr").className = `range-fill ${hr > 100 || hr < 60 ? 'fill-danger' : 'fill-normal'}`;

        document.getElementById("range-bp").textContent = bp;
        document.getElementById("fill-bp").style.width = `${Math.min((bp / 180) * 100, 100)}%`;
        document.getElementById("fill-bp").className = `range-fill ${bp < 90 || bp > 140 ? 'fill-danger' : 'fill-normal'}`;

        document.getElementById("range-spo2").textContent = spo2;
        document.getElementById("fill-spo2").style.width = `${spo2}%`;
        document.getElementById("fill-spo2").className = `range-fill ${spo2 < 94 ? 'fill-danger' : 'fill-normal'}`;
    }

    // Update Risk Contribution Matrix (Pie Chart)
    if(pieChart) {
        let hrWeight = Math.abs(hr - 75) * 2;
        let bpWeight = Math.abs(120 - bp) * 2.5;
        let rrWeight = Math.abs(16 - rr) * 4;
        let tempWeight = Math.abs(37.0 - temp) * 15;
        let otherWeight = marker * 100 + 10;
        pieChart.data.datasets[0].data = [hrWeight, bpWeight, tempWeight, rrWeight, otherWeight];
        pieChart.update();
    }

    // Vitals Summary
    vitalsSummaryList.innerHTML = res.explanation.map(exp => `<li><i data-lucide="check-circle-2"></i> ${exp}</li>`).join("");
    lucide.createIcons();
    
    // AI Clinical Synthesis Generation
    const aiTextEl = document.getElementById("ai-summary-text");
    const aiCardEl = document.getElementById("ai-synthesis-card");
    if(aiTextEl && aiCardEl) {
        if (score < 30) {
            aiCardEl.style.borderLeft = "4px solid var(--success)";
        } else if (score < 70) {
            aiCardEl.style.borderLeft = "4px solid var(--warning)";
        } else {
            aiCardEl.style.borderLeft = "4px solid var(--danger)";
        }
        
        // Grab authentic generation from the Flask/Gemini response!
        let aiSummary = res.ai_synthesis || "Processing delayed, neural engine offline.";
        
        if(window.aiTypewriter) clearTimeout(window.aiTypewriter);
        aiTextEl.textContent = "";
        let i = 0;
        function typeWriter() {
            if (i < aiSummary.length) {
                aiTextEl.textContent += aiSummary.charAt(i);
                i++;
                window.aiTypewriter = setTimeout(typeWriter, 15);
            }
        }
        typeWriter();
    }
    
    // Update Chart
    updateTrend(score, accentColor);
}

function animateCounter(el, target) {
    const start = parseFloat(el.textContent);
    const duration = 800;
    const startTime = performance.now();
    
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const current = start + (target - start) * progress;
        el.textContent = Math.round(current);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function initChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');
    trendChart = new Chart(ctx, {
        type: 'line',
        data: trendData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    beginAtZero: true, 
                    max: 100, 
                    grid: { color: 'rgba(56, 189, 248, 0.05)' }, 
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                x: { display: false }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateTrend(newScore, color) {
    trendChart.data.datasets[0].data.shift();
    trendChart.data.datasets[0].data.push(newScore);
    trendChart.data.datasets[0].borderColor = color;
    trendChart.data.datasets[0].backgroundColor = color + "1a";
    trendChart.update();
}

let pieChart;
function initPieChart() {
    const ctx = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Heart Rate', 'Blood Pressure', 'Temperature', 'Respiration', 'Infection Marker'],
            datasets: [{
                data: [20, 20, 20, 20, 20],
                backgroundColor: ['#ef4444', '#f59e0b', '#38bdf8', '#818cf8', '#10b981'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#94a3b8', font: { size: 10 } }
                }
            }
        }
    });
}
