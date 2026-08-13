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
                    Infection_Marker: parseFloat(document.getElementById("marker").value || 0),
                    generate_synthesis: false
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
        Infection_Marker: parseFloat(document.getElementById("marker").value),
        generate_synthesis: true
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

const API_KEY = 'sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b';

function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Prediction Logic
async function getPrediction(vitals) {
    const btn = document.getElementById("predict-btn");
    btn.classList.add("btn-loading");
    
    try {
        const resp = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
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
    if (score >= 50 || res.alert_level === 'CRITICAL') {
        accentColor = "#ef4444"; // Danger
        gaugeFillEl.classList.add("critical-glow");
        alertBox.classList.remove("hidden");
        if (window.telemetryAudio) window.telemetryAudio.playBeep('CRITICAL');
    } else if (score >= 27 || res.alert_level === 'WARNING') {
        accentColor = "#f59e0b"; // Warning
        gaugeFillEl.classList.remove("critical-glow");
        alertBox.classList.add("hidden");
        if (window.telemetryAudio) window.telemetryAudio.playBeep('WARNING');
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
    
    const sirsEl = document.getElementById("sirs-score");
    if (sirsEl) sirsEl.textContent = `${res.sirs_score ?? 0}/4`;
    const qsofaEl = document.getElementById("qsofa-score");
    if (qsofaEl) qsofaEl.textContent = `${res.qsofa_score ?? 0}/2`;

    // Update Ranges UI
    const bp = parseFloat(document.getElementById("bp").value);
    const spo2 = parseFloat(document.getElementById("spo2").value);
    const age = parseInt(document.getElementById("age").value);
    
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
        
        document.getElementById("range-temp").textContent = temp;
        document.getElementById("fill-temp").style.width = `${Math.max(0, Math.min(((temp - 30) / 15) * 100, 100))}%`;
        document.getElementById("fill-temp").className = `range-fill ${temp < 36 || temp > 38 ? 'fill-danger' : 'fill-normal'}`;
        
        document.getElementById("range-rr").textContent = rr;
        document.getElementById("fill-rr").style.width = `${Math.min((rr / 40) * 100, 100)}%`;
        document.getElementById("fill-rr").className = `range-fill ${rr > 20 || rr < 12 ? 'fill-danger' : 'fill-normal'}`;

        document.getElementById("range-age").textContent = age;
        document.getElementById("fill-age").style.width = `${Math.min((age / 100) * 100, 100)}%`;
        document.getElementById("fill-age").className = `range-fill fill-normal`;

        document.getElementById("range-marker").textContent = marker.toFixed(2);
        document.getElementById("fill-marker").style.width = `${Math.min(marker * 100, 100)}%`;
        document.getElementById("fill-marker").className = `range-fill ${marker > 0.5 ? 'fill-danger' : 'fill-normal'}`;
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
    vitalsSummaryList.innerHTML = (res.explanation || []).map(exp => `<li><i data-lucide="check-circle-2"></i> ${escapeHtml(exp)}</li>`).join("");
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
    const canvas = document.getElementById('trendChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 0, 150);
    gradient.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
    gradient.addColorStop(0.5, 'rgba(56, 189, 248, 0.12)');
    gradient.addColorStop(1, 'rgba(56, 189, 248, 0.01)');

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(10).fill('').map((_, i) => i === 9 ? 'NOW' : `-${(9 - i) * 2}s`),
            datasets: [{
                label: 'Sepsis Risk (%)',
                data: Array(10).fill(15),
                borderColor: '#38bdf8',
                borderWidth: 2.5,
                backgroundColor: gradient,
                fill: true,
                tension: 0.35,
                pointRadius: (c) => (c.dataIndex === 9 ? 5 : 2),
                pointHoverRadius: 6,
                pointBackgroundColor: (c) => (c.dataIndex === 9 ? '#ffffff' : '#38bdf8'),
                pointBorderColor: '#38bdf8',
                pointBorderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { 
                    beginAtZero: true, 
                    max: 100, 
                    grid: { color: 'rgba(255, 255, 255, 0.06)', drawBorder: false }, 
                    ticks: { color: '#94a3b8', font: { size: 9, family: 'monospace' }, callback: (v) => v + '%' }
                },
                x: {
                    ticks: { color: '#64748b', font: { size: 9, family: 'monospace' } },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10,15,25,0.92)',
                    titleColor: '#38bdf8',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(56,189,248,0.3)',
                    borderWidth: 1,
                    padding: 8,
                    displayColors: false,
                    callbacks: {
                        label: (c) => `Risk Score: ${c.parsed.y.toFixed(1)}%`
                    }
                }
            }
        }
    });
}

function updateTrend(newScore, color) {
    if (!trendChart) return;
    const colorHex = color || '#38bdf8';
    trendChart.data.datasets[0].data.shift();
    trendChart.data.datasets[0].data.push(newScore);
    trendChart.data.datasets[0].borderColor = colorHex;
    trendChart.data.datasets[0].pointBorderColor = colorHex;
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
