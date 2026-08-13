/**
 * SepsisGuard AI v3.0 — Patient Detail Screen
 * Real-time single-patient monitoring with animated risk orb
 */

const SERVER  = window.location.origin;
const API_KEY = 'sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b';
const pid     = new URLSearchParams(window.location.search).get('pid') || 'P001';

function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

let trendChart = null;
let ecgRenderer = null;
let orbAnim    = null;
let currentRisk = 0;
let currentColor = '#10b981';
let currentLevel = 'STABLE';
let prevLevel = null;
let currentVitals = {};
let hasAssessed = false;

// ─── CLINICIAN RISK ASSESSMENT WORKFLOW ─────────────────────────
function runClinicianAssessment() {
    const emptyState = document.getElementById('empty-assessment-state');
    const loadingState = document.getElementById('loading-assessment-state');
    const errorBanner = document.getElementById('assessment-error-banner');
    const tsEl = document.getElementById('assessment-timestamp');
    const btn = document.getElementById('run-assessment-btn');

    if (emptyState) emptyState.style.display = 'none';
    if (errorBanner) errorBanner.style.display = 'none';
    if (loadingState) loadingState.style.display = 'block';
    if (btn) { btn.disabled = true; btn.innerHTML = '<span>⏳</span><span>Analyzing Patient Data...</span>'; }

    const payload = {
        Heart_Rate: currentVitals.Heart_Rate || 80,
        Oxygen_Level: currentVitals.Oxygen_Level || 98,
        Temperature: currentVitals.Temperature || 37.0,
        Blood_Pressure: currentVitals.Blood_Pressure || 120,
        Resp_Rate: currentVitals.Resp_Rate || 16,
        Infection_Marker: currentVitals.Infection_Marker || 0.5,
        Age: currentVitals.Age || 65,
        generate_synthesis: true
    };

    fetch(SERVER + '/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
        },
        body: JSON.stringify(payload)
    })
    .then(async (res) => {
        if (!res.ok) {
            if (res.status === 401) throw new Error("Your session or API authorization is no longer valid.");
            if (res.status === 422) throw new Error("Some patient measurements are invalid or missing.");
            if (res.status === 429) throw new Error("Too many requests. Please wait before trying again.");
            throw new Error("Risk assessment is temporarily unavailable.");
        }
        return res.json();
    })
    .then((data) => {
        if (loadingState) loadingState.style.display = 'none';
        if (btn) { btn.disabled = false; btn.innerHTML = '<span>⚡</span><span>Run Risk Assessment</span>'; }
        hasAssessed = true;

        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        if (tsEl) { tsEl.style.display = 'block'; tsEl.textContent = 'Last assessed: ' + timeStr; }

        renderAll({ ...data, vitals: payload, name: document.getElementById('pt-name')?.textContent || pid });
    })
    .catch((err) => {
        if (loadingState) loadingState.style.display = 'none';
        if (btn) { btn.disabled = false; btn.innerHTML = '<span>⚡</span><span>Run Risk Assessment</span>'; }
        if (errorBanner) {
            errorBanner.style.display = 'flex';
            const msgEl = document.getElementById('assessment-error-msg');
            if (msgEl) msgEl.textContent = err.message || "Risk assessment could not be completed. Please verify patient data and try again.";
        }
    });
}

// ─── Socket.IO ──────────────────────────────────
const socket = io(SERVER, {
    auth: { token: API_KEY },
    transports: ['websocket', 'polling']
});

socket.on('connect', () => setConn(true));
socket.on('disconnect', () => setConn(false));

socket.on('snapshot', (data) => {
    if (data[pid]) renderAll(data[pid]);
});

socket.on('telemetry', (pkt) => {
    if (pkt.pid === pid) renderAll(pkt);
});

socket.on('ai_synthesis_result', (data) => {
    if (data.pid === pid && data.ai_synthesis) {
        typewriter(document.getElementById('pt-synthesis'), data.ai_synthesis);
        const btn = document.getElementById('pt-ai-btn');
        btn.textContent = '⚡ Generate AI Synthesis';
        btn.disabled = false;
    }
});

socket.on('timeline_event', (data) => {
    if (data.pid === pid) addEvent(data.event);
});

socket.on('timeline_snapshot', (data) => {
    if (data[pid]) data[pid].forEach(e => addEvent(e));
});

// ─── Render everything ──────────────────────────
function renderAll(data) {
    const v     = data.vitals || data;
    const risk  = data.risk_score ?? 0;
    const color = data.risk_color  || '#10b981';
    const level = data.alert_level || 'STABLE';

    currentRisk  = risk;
    currentColor = color;
    currentLevel = level;

    // Audio beeping on alert escalation
    if (level === 'CRITICAL' && prevLevel !== 'CRITICAL') {
        if (window.telemetryAudio) window.telemetryAudio.playBeep('CRITICAL');
    } else if (level === 'WARNING' && prevLevel !== 'WARNING' && prevLevel !== 'CRITICAL') {
        if (window.telemetryAudio) window.telemetryAudio.playBeep('WARNING');
    }
    prevLevel = level;

    // Header
    setText('pt-name',  data.name  || pid);
    setText('pt-meta',  (data.bed || '—') + ' · Age ' + (data.age || '—') + ' · ' + (data.room || '—'));
    const badge = document.getElementById('pt-badge');
    if (badge) { badge.className = 'alert-badge ' + level; badge.textContent = level; }

    // Vitals
    setVCard('v-hr',   'vb-hr',   v.Heart_Rate,       60,  100,  false, 'bpm');
    setVCard('v-bp',   'vb-bp',   v.Blood_Pressure,   90,  120,  false, 'mmHg');
    setVCard('v-spo2', 'vb-spo2', v.Oxygen_Level,     95,  100,  false, '%');
    setVCard('v-temp', 'vb-temp', v.Temperature,      36.5,37.5, false, '°C');
    setVCard('v-rr',   'vb-rr',   v.Resp_Rate,        12,  20,   false, 'bpm');
    setVCard('v-inf',  'vb-inf',  v.Infection_Marker, 0,   0.5,  true,  '');

    currentVitals = v;

    // Orb
    setText('orb-pct',   risk.toFixed(0) + '%');
    setText('orb-level', data.risk_level || '—');
    const lvlEl = document.getElementById('orb-level');
    if (lvlEl) lvlEl.style.color = color;

    // Metrics
    setText('m-sirs',  (data.sirs_score ?? '—') + '/4');
    setText('m-qsofa', (data.qsofa_score ?? '—') + '/2');
    updateCriteriaChecklist(v, data);

    // Triggers
    const trig = document.getElementById('pt-triggers');
    if (trig) {
        const exp = data.explanation || [];
        trig.innerHTML = exp.length
            ? exp.map(e => `<span class="trigger-tag">${e}</span>`).join('')
            : '<span class="trigger-tag ok">All vitals within normal range</span>';
    }

    // AI Synthesis
    if (data.ai_synthesis) setText('pt-synthesis', data.ai_synthesis);

    // Contributions
    renderContribs(data.contributions || {});

    // Trend
    updateTrend(data.trend || [], color);

    // ECG
    if (ecgRenderer) {
        if (v.Heart_Rate) ecgRenderer.setHR(v.Heart_Rate);
        ecgRenderer.setLevel(level);
    }

    // Alert glow on body
    document.body.style.boxShadow = level === 'CRITICAL'
        ? 'inset 0 0 80px rgba(239,68,68,0.08)'
        : 'none';
}

// ─── Criteria Checklist ─────────────────────────
function updateCriteriaChecklist(v, data) {
    if (!v) return;
    const sirsC = data.sirs_criteria || {
        temp_met: v.Temperature < 36 || v.Temperature > 38,
        hr_met: v.Heart_Rate > 90,
        rr_met: v.Resp_Rate > 20,
        wbc_met: v.Infection_Marker > 0.5
    };
    const qsofaC = data.qsofa_criteria || {
        rr_met: v.Resp_Rate >= 22,
        sbp_met: v.Blood_Pressure <= 100
    };

    const setChk = (id, met) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.className = 'chk-item ' + (met ? 'met' : 'unmet');
        const bdg = el.querySelector('.chk-badge');
        if (bdg) bdg.textContent = met ? '✓ Criterion Met' : '○ Not Met';
    };

    setChk('chk-temp', sirsC.temp_met);
    setChk('chk-hr', sirsC.hr_met);
    setChk('chk-rr', sirsC.rr_met);
    setChk('chk-wbc', sirsC.wbc_met);
    setChk('chk-qrr', qsofaC.rr_met);
    setChk('chk-sbp', qsofaC.sbp_met);
}

// ─── Vital Card ─────────────────────────────────
function setVCard(valId, barId, val, lo, hi, invert, unit) {
    const el  = document.getElementById(valId);
    const bar = document.getElementById(barId);
    if (!el || val == null) return;

    const fmt = unit === '°C' ? val.toFixed(1) : val > 1 ? Math.round(val) : val.toFixed(2);
    el.textContent = fmt + (unit ? ' ' + unit : '');

    let cls = '';
    if (invert) { if (val >= hi) cls = 'danger'; }
    else if (val < lo || val > hi) { cls = 'danger'; }

    el.className = 'vcard-val ' + cls;
    if (bar) {
        const pct = invert
            ? Math.min(val / hi * 100, 100)
            : Math.min(((val - lo) / (hi - lo + 0.001)) * 100, 100);
        bar.style.width   = Math.max(2, pct) + '%';
        bar.style.background = cls === 'danger' ? '#ef4444' : '#10b981';
    }
}

// ─── Model Explanation (SHAP) ───────────────────
function renderContribs(shap_data) {
    const el = document.getElementById('contrib-bars');
    if (!el) return;
    el.innerHTML = '';

    if (!shap_data || shap_data.available === false) {
        el.innerHTML = '<div style="font-size:0.7rem;color:var(--text-2);padding:6px">Model explanation is temporarily unavailable.</div>';
        return;
    }

    const features = shap_data.features || [];
    if (features.length === 0) {
        el.innerHTML = '<div style="font-size:0.7rem;color:var(--text-2);padding:6px">No significant feature attributions.</div>';
        return;
    }

    // Subheader
    const subheader = document.createElement('div');
    subheader.style.cssText = 'font-size:0.62rem;color:var(--accent);font-family:var(--mono);margin-bottom:8px;display:flex;justify-content:space-between;align-items:center';
    subheader.innerHTML = '<span>&larr; Decreases Risk</span><span>Model Explanation — SHAP</span><span>Increases Risk &rarr;</span>';
    el.appendChild(subheader);

    // Max absolute SHAP value for scaling bars (max 45% width on either side of 50% center)
    const maxShap = Math.max(...features.map(f => Math.abs(f.shap_value || 0)), 0.05);

    features.forEach(f => {
        const name = f.display_name || f.feature;
        const valStr = f.value != null ? (f.unit ? `${f.value} ${f.unit}` : `${f.value}`) : '';
        const shapVal = f.shap_value || 0;
        const isPos = shapVal >= 0;
        const widthPct = Math.min(Math.round((Math.abs(shapVal) / maxShap) * 45), 45);
        
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;flex-direction:column;gap:2px;margin-bottom:6px;background:rgba(255,255,255,0.02);padding:6px;border-radius:4px;font-size:0.7rem';

        const signStr = isPos ? `+${shapVal.toFixed(3)}` : shapVal.toFixed(3);
        const dirArrow = isPos ? '↑' : '↓';
        const barColor = isPos ? '#ef4444' : '#10b981';

        row.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;font-weight:600">
                <span style="color:var(--text-1)">${name} <span style="font-weight:400;color:var(--text-2);font-size:0.63rem">(${valStr})</span></span>
                <span style="color:${barColor};font-family:var(--mono)">${dirArrow} ${signStr}</span>
            </div>
            <div style="position:relative;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;margin-top:3px">
                <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.25);z-index:2"></div>
                <div style="position:absolute;top:0;bottom:0;background:${barColor};border-radius:2px;${isPos ? `left:50%;width:${widthPct}%` : `right:50%;width:${widthPct}%`}"></div>
            </div>
            <div style="font-size:0.62rem;color:var(--text-2);margin-top:2px">${f.formatted_text || ''}</div>
        `;
        el.appendChild(row);
    });
}

// ─── Trend Chart ────────────────────────────────
function updateTrend(trend, color) {
    const canvas = document.getElementById('pt-trend');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const colorHex = color || '#38bdf8';

    const labels = trend.map((_, i) => {
        const pastSec = (trend.length - 1 - i) * 2;
        return pastSec === 0 ? 'NOW' : `-${pastSec}s`;
    });

    if (!trendChart) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 150);
        gradient.addColorStop(0, colorHex + '55');
        gradient.addColorStop(0.5, colorHex + '22');
        gradient.addColorStop(1, colorHex + '03');

        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sepsis Risk (%)',
                    data: trend,
                    borderColor: colorHex,
                    borderWidth: 2.5,
                    backgroundColor: gradient,
                    fill: true,
                    tension: 0.35,
                    pointRadius: (c) => (c.dataIndex === c.dataset.data.length - 1 ? 5 : 2),
                    pointHoverRadius: 6,
                    pointBackgroundColor: (c) => (c.dataIndex === c.dataset.data.length - 1 ? '#ffffff' : colorHex),
                    pointBorderColor: colorHex,
                    pointBorderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 25,
                            color: '#94a3b8',
                            font: { size: 9, family: 'monospace' },
                            callback: (v) => v + '%'
                        },
                        grid: { color: 'rgba(255,255,255,0.06)', drawBorder: false }
                    },
                    x: {
                        ticks: {
                            color: '#64748b',
                            font: { size: 9, family: 'monospace' },
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 7
                        },
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
    } else {
        trendChart.data.labels = labels;
        trendChart.data.datasets[0].data = trend;
        trendChart.data.datasets[0].borderColor = colorHex;
        trendChart.data.datasets[0].pointBorderColor = colorHex;
        trendChart.update('none');
    }
}

// ─── Timeline ───────────────────────────────────
function addEvent(evt) {
    const tl  = document.getElementById('pt-timeline');
    if (!tl) return;
    const div = document.createElement('div');
    div.className = 'tl-event ' + (evt.type || 'TRIGGER');
    div.innerHTML = `<div class="tl-time">${evt.time}</div><div class="tl-msg">${evt.msg}</div>`;
    tl.prepend(div);
    while (tl.children.length > 25) tl.lastChild.remove();
}

// ─── AI Request ─────────────────────────────────
function requestAI() {
    const btn = document.getElementById('pt-ai-btn');
    btn.textContent = '⚡ Generating…'; btn.disabled = true;
    setText('pt-synthesis', 'AI clinical synthesis generating…');
    socket.emit('request_ai_synthesis', { pid });
}

// ─── Risk Orb Canvas ────────────────────────────
function startOrb() {
    const canvas = document.getElementById('orb-canvas');
    const ctx    = canvas.getContext('2d');
    const W = 200, H = 200, cx = 100, cy = 100;
    let angle = 0;

    function draw() {
        ctx.clearRect(0, 0, W, H);
        const risk  = currentRisk;
        const color = currentColor;
        const level = currentLevel;

        // Outer ring (full circle background)
        ctx.beginPath();
        ctx.arc(cx, cy, 85, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(56,189,248,0.07)';
        ctx.lineWidth   = 8;
        ctx.stroke();

        // Risk arc
        const startA = -Math.PI / 2;
        const endA   = startA + (risk / 100) * Math.PI * 2;
        ctx.shadowColor = color; ctx.shadowBlur = 18;
        ctx.beginPath();
        ctx.arc(cx, cy, 85, startA, endA);
        ctx.strokeStyle = color;
        ctx.lineWidth   = 8;
        ctx.lineCap     = 'round';
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Rotating outer ring (scanner effect)
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);
        const grad = ctx.createLinearGradient(0, -95, 0, 95);
        grad.addColorStop(0,   color + '00');
        grad.addColorStop(0.5, color + '40');
        grad.addColorStop(1,   color + '00');
        ctx.beginPath();
        ctx.arc(0, 0, 95, -0.3, 0.3);
        ctx.strokeStyle = grad;
        ctx.lineWidth   = 3;
        ctx.stroke();
        ctx.restore();

        // Inner glow circle
        const pulse = 0.7 + Math.sin(angle * 3) * 0.08;
        const radialGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 65 * pulse);
        radialGrad.addColorStop(0,   color + '20');
        radialGrad.addColorStop(0.6, color + '08');
        radialGrad.addColorStop(1,   'transparent');
        ctx.beginPath();
        ctx.arc(cx, cy, 65 * pulse, 0, Math.PI * 2);
        ctx.fillStyle = radialGrad;
        ctx.fill();

        // Middle ring
        ctx.beginPath();
        ctx.arc(cx, cy, 65, 0, Math.PI * 2);
        ctx.strokeStyle = color + '20';
        ctx.lineWidth   = 1;
        ctx.stroke();

        // Tick marks for critical
        if (level === 'CRITICAL') {
            for (let i = 0; i < 8; i++) {
                const a = (i / 8) * Math.PI * 2 + angle;
                const r1 = 74, r2 = 80;
                ctx.beginPath();
                ctx.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
                ctx.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
                ctx.strokeStyle = color + 'aa';
                ctx.lineWidth   = 1.5;
                ctx.stroke();
            }
        }

        angle += level === 'CRITICAL' ? 0.035 : 0.018;
        orbAnim = requestAnimationFrame(draw);
    }
    draw();
}

// ─── ECG Renderer (same as dashboard.js) ────────
class ECGRenderer {
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');
        this.speed  = opts.speed || 3;
        this.lw     = opts.lineWidth || 2;
        this.hr     = 75; this.color = '#10b981';
        this.phase  = 0; this.buf = []; this.on = false; this._raf = null;
        this._resize();
        new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    }
    _resize() {
        const p = this.canvas.parentElement;
        if (!p) return;
        this.canvas.width  = p.clientWidth;
        this.canvas.height = p.clientHeight;
        this.W = this.canvas.width; this.H = this.canvas.height;
        this.buf = new Array(this.W).fill(this.H / 2);
    }
    setHR(hr)  { this.hr = Math.max(30, Math.min(200, hr)); }
    setLevel(l){ this.color = l === 'CRITICAL' ? '#ef4444' : l === 'WARNING' ? '#f59e0b' : '#10b981'; }
    start()    { this.on = true; this._render(); }
    stop()     { this.on = false; if (this._raf) cancelAnimationFrame(this._raf); }
    _sample(phase) {
        const t = ((phase % (Math.PI * 2)) / (Math.PI * 2));
        if (t < 0.13) return 0.12 * Math.sin((t / 0.13) * Math.PI);
        if (t < 0.21) return 0;
        if (t < 0.23) return -0.12 * (t - 0.21) / 0.02;
        if (t < 0.26) return  1.0  * (t - 0.23) / 0.03;
        if (t < 0.29) return  1.0  * (1 - (t - 0.26) / 0.03);
        if (t < 0.32) return -0.14 * (1 - (t - 0.29) / 0.03);
        if (t < 0.44) return 0;
        if (t < 0.63) return 0.24 * Math.sin(((t - 0.44) / 0.19) * Math.PI);
        return 0;
    }
    _render() {
        if (!this.on) return;
        const { ctx, W, H, speed } = this;
        if (!W || !H) { this._raf = requestAnimationFrame(() => this._render()); return; }
        const midY = H * 0.5, amp = H * 0.38;
        const inc  = (this.hr / 60) * (Math.PI * 2) / 60 * speed;
        for (let s = 0; s < speed; s++) {
            this.phase += inc / speed;
            this.buf.push(midY - this._sample(this.phase) * amp + (Math.random() - 0.5) * 0.5);
        }
        if (this.buf.length > W + speed) this.buf.splice(0, this.buf.length - W);
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = 'rgba(56,189,248,0.04)'; ctx.lineWidth = 0.5;
        for (let gy = 0.25; gy < 1; gy += 0.25) { ctx.beginPath(); ctx.moveTo(0, H * gy); ctx.lineTo(W, H * gy); ctx.stroke(); }
        ctx.save();
        ctx.shadowColor = this.color; ctx.shadowBlur = 10;
        ctx.strokeStyle = this.color; ctx.lineWidth = this.lw; ctx.lineJoin = 'round';
        ctx.beginPath();
        const st = Math.max(0, this.buf.length - W);
        for (let x = 0; x < Math.min(this.buf.length, W); x++) {
            const y = this.buf[st + x];
            x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.restore();
        this._raf = requestAnimationFrame(() => this._render());
    }
}

// ─── Utilities ──────────────────────────────────
function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }
function setConn(on) {
    const dot = document.getElementById('conn-dot'), lbl = document.getElementById('conn-lbl');
    if (dot) dot.className = 'sdot' + (on ? '' : ' off');
    if (lbl) { lbl.textContent = on ? 'Live' : 'Disconnected'; lbl.style.color = on ? '#10b981' : '#ef4444'; }
}
function typewriter(el, text) {
    if (!el) return; el.textContent = ''; let i = 0;
    const t = setInterval(() => { if (i < text.length) el.textContent += text[i++]; else clearInterval(t); }, 14);
}
setInterval(() => { setText('clock', new Date().toLocaleTimeString('en-GB', { hour12: false })); }, 1000);

// ─── Init ────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    // Update page title
    document.title = 'SepsisGuard — Patient ' + pid;

    // ECG
    const ecgCanvas = document.getElementById('pt-ecg');
    ecgRenderer = new ECGRenderer(ecgCanvas, { speed: 3, lineWidth: 2 });
    ecgRenderer.start();

    // Risk Orb
    startOrb();

    // Initial Fetch via REST so page populates immediately without waiting for telemetry broadcast
    fetch(SERVER + '/patients', {
        headers: { 'X-API-Key': API_KEY }
    })
    .then(res => res.json())
    .then(data => {
        if (data && data[pid]) {
            renderAll(data[pid]);
        }
    })
    .catch(err => console.error('[Patient Init Fetch Error]', err));
});

// Update dashboard.js card link
// Make patient cards clickable to this page
if (window.parent && window.parent !== window) {
    // If embedded, communicate via postMessage
}
