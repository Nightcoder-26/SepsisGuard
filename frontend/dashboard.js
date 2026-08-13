/**
 * SepsisGuard AI v3.0 — ICU Intelligence Ecosystem
 * Part 1: Core systems — Socket.IO, Cards, ECG, Triage, Timeline
 */

const SERVER  = 'http://localhost:5000';
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

// ─── State ────────────────────────────────────────
const patients      = {};
const ecgRenderers  = {};
const prevAlerts    = {};
let   pktCount      = 0;
let   modalPid      = null;
let   modalEcg      = null;
let   modalChart    = null;
let   selectedPid   = null; // for explainability panel

// ─── Socket.IO ────────────────────────────────────
const socket = io(SERVER, {
    auth: { token: API_KEY },
    transports: ['websocket', 'polling']
});

socket.on('connect', () => {
    setConn(true);
    console.log('[SepsisGuard v3] Connected');
});
socket.on('disconnect', () => setConn(false));

socket.on('snapshot', (data) => {
    for (const [pid, state] of Object.entries(data)) {
        patients[pid] = state;
        if (!document.getElementById('card-' + pid)) createCard(pid, state);
        updateCard(pid, state);
    }
    renderTriage();
    updateStatusBar();
    if (!selectedPid && Object.keys(patients).length) {
        selectedPid = Object.keys(patients)[0];
        renderExplain(selectedPid);
    }
});

socket.on('timeline_snapshot', (data) => {
    for (const [pid, events] of Object.entries(data)) {
        for (const evt of events) addTimelineEvent(pid, evt, false);
    }
});

socket.on('telemetry', (pkt) => {
    pktCount++;
    document.getElementById('pkt-ctr').textContent = 'PKT: ' + pktCount.toLocaleString();

    const pid = pkt.pid;
    patients[pid] = { ...(patients[pid] || {}), ...pkt };

    if (!document.getElementById('card-' + pid)) createCard(pid, pkt);
    updateCard(pid, pkt);
    updateStatusBar();
    renderTriage();

    if (selectedPid === pid) renderExplain(pid);
    if (modalPid === pid) updateModal(pkt);

    // Alert toast & medical sound beeps on alert state escalation
    const prev = prevAlerts[pid];
    if (pkt.alert_level === 'CRITICAL' && prev !== 'CRITICAL') {
        fireToast(pkt, 'CRITICAL');
        if (window.telemetryAudio) window.telemetryAudio.playBeep('CRITICAL');
    } else if (pkt.alert_level === 'WARNING' && prev !== 'WARNING' && prev !== 'CRITICAL') {
        fireToast(pkt, 'WARNING');
        if (window.telemetryAudio) window.telemetryAudio.playBeep('WARNING');
    }
    prevAlerts[pid] = pkt.alert_level;

    // Anomaly flash
    if (pkt.anomaly) {
        const card = document.getElementById('card-' + pid);
        if (card) { card.classList.add('anomaly'); setTimeout(() => card.classList.remove('anomaly'), 500); }
    }

    floorMap.update();
});

socket.on('timeline_event', (data) => {
    addTimelineEvent(data.pid, data.event, true);
});

socket.on('ai_synthesis_result', (data) => {
    if (data.pid && patients[data.pid]) {
        patients[data.pid].ai_synthesis = data.ai_synthesis;
        updateCard(data.pid, patients[data.pid]);
        if (modalPid === data.pid) {
            typewriter(document.getElementById('m-synthesis'), data.ai_synthesis);
            document.getElementById('m-ai-btn').textContent = '⚡ Generate Full AI Synthesis';
            document.getElementById('m-ai-btn').disabled = false;
        }
    }
});

socket.on('copilot_response', (data) => {
    removeCopilotThinking();
    appendCopilotMsg(data.answer, 'ai');
});

// ─── Connection Status ─────────────────────────────
function setConn(on) {
    const dot = document.getElementById('conn-dot');
    const lbl = document.getElementById('conn-lbl');
    dot.className = 'sdot' + (on ? '' : ' off');
    lbl.textContent = on ? 'Neural Node Online' : 'Disconnected';
    lbl.style.color = on ? '#10b981' : '#ef4444';
}

// ─── Status Bar ───────────────────────────────────
function updateStatusBar() {
    let c = 0, w = 0, s = 0;
    for (const pid of Object.keys(patients)) {
        const l = patients[pid]?.alert_level;
        if (l === 'CRITICAL') c++; else if (l === 'WARNING') w++; else s++;
    }
    document.getElementById('sv-crit').textContent = c;
    document.getElementById('sv-warn').textContent = w;
    document.getElementById('sv-ok').textContent   = s;

    const smCrit = document.getElementById('sm-crit'); if (smCrit) smCrit.textContent = c;
    const smWarn = document.getElementById('sm-warn'); if (smWarn) smWarn.textContent = w;
    const smOk = document.getElementById('sm-ok'); if (smOk) smOk.textContent = s;
    const smTotal = document.getElementById('sm-total'); if (smTotal) smTotal.textContent = Object.keys(PATIENTS).length;
}

let currentRiskFilter = 'ALL';

function setRiskFilter(filter, el) {
    currentRiskFilter = filter;
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    if (el) el.classList.add('active');
    filterPatients();
}

function filterPatients() {
    const q = (document.getElementById('patient-search')?.value || '').toLowerCase();
    for (const pid of Object.keys(PATIENTS)) {
        const card = document.getElementById('card-' + pid);
        if (!card) continue;
        const p = patients[pid];
        const level = p?.alert_level || 'STABLE';
        const name = (p?.name || '').toLowerCase();
        const bed = (p?.bed || '').toLowerCase();

        const matchesSearch = !q || name.includes(q) || bed.includes(q) || pid.toLowerCase().includes(q);
        const matchesFilter = currentRiskFilter === 'ALL' || level === currentRiskFilter;

        card.style.display = (matchesSearch && matchesFilter) ? 'flex' : 'none';
    }
}

function openAboutModelModal() {
    const modal = document.getElementById('about-model-modal');
    if (modal) modal.classList.add('open');
}

function closeAboutModelModal() {
    const modal = document.getElementById('about-model-modal');
    if (modal) modal.classList.remove('open');
}

// ─── Clock ────────────────────────────────────────
setInterval(() => {
    document.getElementById('clock').textContent =
        new Date().toLocaleTimeString('en-GB', { hour12: false });
}, 1000);

// ─── Patient Cards ────────────────────────────────
function createCard(pid, data) {
    const grid = document.getElementById('patient-grid');
    const card = document.createElement('div');
    card.id        = 'card-' + pid;
    card.className = 'patient-card ' + (data.alert_level || 'STABLE');
    card.onclick   = (e) => { if (!e.ctrlKey && !e.metaKey) { openModal(pid); } };
    card.ondblclick = () => window.open('/patient?pid=' + pid, '_blank');
    card.innerHTML = `
        <div class="card-header">
            <div>
                <div class="patient-name" id="pn-${escapeHtml(pid)}">${escapeHtml(data.name || pid)}</div>
                <div class="patient-meta" id="pm-${escapeHtml(pid)}">${escapeHtml(data.bed || '—')} · Age ${escapeHtml(data.age || '—')}</div>
            </div>
            <div class="alert-badge ${escapeHtml(data.alert_level || 'STABLE')}" id="pb-${escapeHtml(pid)}">${escapeHtml(data.alert_level || 'STABLE')}</div>
        </div>
        <div class="ecg-strip">
            <span class="ecg-lbl">ECG II</span>
            <canvas class="ecg-canvas" id="ecg-${escapeHtml(pid)}"></canvas>
        </div>
        <div class="vitals-mini">
            <div class="vc" id="vc-hr-${escapeHtml(pid)}"><div class="vval" id="vv-hr-${escapeHtml(pid)}">—</div><div class="vlbl">HR bpm</div></div>
            <div class="vc" id="vc-bp-${escapeHtml(pid)}"><div class="vval" id="vv-bp-${escapeHtml(pid)}">—</div><div class="vlbl">SysBP</div></div>
            <div class="vc" id="vc-spo2-${escapeHtml(pid)}"><div class="vval" id="vv-spo2-${escapeHtml(pid)}">—</div><div class="vlbl">SpO₂%</div></div>
            <div class="vc" id="vc-temp-${escapeHtml(pid)}"><div class="vval" id="vv-temp-${escapeHtml(pid)}">—</div><div class="vlbl">Temp°C</div></div>
            <div class="vc" id="vc-rr-${escapeHtml(pid)}"><div class="vval" id="vv-rr-${escapeHtml(pid)}">—</div><div class="vlbl">RR bpm</div></div>
            <div class="vc" id="vc-inf-${escapeHtml(pid)}"><div class="vval" id="vv-inf-${escapeHtml(pid)}">—</div><div class="vlbl">InfMkr</div></div>
        </div>
        <div class="risk-row">
            <div class="risk-lbl">RISK</div>
            <div class="risk-track"><div class="risk-fill" id="rf-${escapeHtml(pid)}" style="width:0%;background:#10b981"></div></div>
            <div class="risk-pct" id="rp-${escapeHtml(pid)}">0%</div>
        </div>
        <div class="metrics-row" style="grid-template-columns:1fr">
            <div class="mc"><div class="mc-val" id="mc-sirs-${escapeHtml(pid)}">—</div><div class="mc-lbl">SIRS (Rule-based)</div></div>
        </div>
        <div class="synth-snip" id="ss-${escapeHtml(pid)}">Initializing AI…</div>
    `;
    grid.appendChild(card);

    // Start ECG
    const canvas = document.getElementById('ecg-' + pid);
    ecgRenderers[pid] = new ECGRenderer(canvas, { speed: 2, lineWidth: 1.5 });
    ecgRenderers[pid].start();
}

function updateCard(pid, data) {
    const card = document.getElementById('card-' + pid);
    if (!card) return;

    const v     = data.vitals || data;
    const risk  = data.risk_score ?? 0;
    const level = data.alert_level || 'STABLE';
    const color = data.risk_color  || '#10b981';

    card.className = 'patient-card ' + level;

    const badge = document.getElementById('pb-' + pid);
    if (badge) { badge.className = 'alert-badge ' + level; badge.textContent = level; }

    setVital(pid, 'hr',   v.Heart_Rate,       60,  100);
    setVital(pid, 'bp',   v.Blood_Pressure,   90,  120);
    setVital(pid, 'spo2', v.Oxygen_Level,     95,  100);
    setVital(pid, 'temp', v.Temperature,      36.5, 37.5);
    setVital(pid, 'rr',   v.Resp_Rate,        12,  20);
    setVital(pid, 'inf',  v.Infection_Marker,  0,   0.5, true);

    const rf = document.getElementById('rf-' + pid);
    const rp = document.getElementById('rp-' + pid);
    if (rf) { rf.style.width = Math.min(risk, 100) + '%'; rf.style.background = color; }
    if (rp) { rp.textContent = risk.toFixed(0) + '%'; rp.style.color = color; }

    setText('mc-sirs-'  + pid, (data.sirs_score ?? '—') + '/4');

    if (data.ai_synthesis) {
        const ss = document.getElementById('ss-' + pid);
        if (ss) ss.textContent = data.ai_synthesis;
    }

    // ECG modulation
    const ecg = ecgRenderers[pid];
    if (ecg) {
        if (v.Heart_Rate) ecg.setHR(v.Heart_Rate);
        ecg.setLevel(level);
    }
}

function setVital(pid, key, val, lo, hi, invert) {
    const el   = document.getElementById('vv-' + key + '-' + pid);
    const chip = document.getElementById('vc-' + key + '-' + pid);
    if (!el || val == null) return;

    const fmt = key === 'inf' ? val.toFixed(2) : key === 'temp' ? val.toFixed(1) : Math.round(val);
    el.textContent = fmt;

    let cls = '';
    if (invert)          { if (val >= hi) cls = 'danger'; }
    else if (val < lo || val > hi) { cls = val < lo * 0.85 || val > hi * 1.15 ? 'danger' : 'warning'; }

    el.className   = 'vval ' + cls;
    chip.className = 'vc '   + cls;
}

function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }

// ─── AI TRIAGE ENGINE ──────────────────────────────
function renderTriage() {
    const list = document.getElementById('triage-list');
    if (!list) return;

    const sorted = Object.entries(patients)
        .map(([pid, d]) => ({ pid, name: d.name, risk: d.risk_score || 0, color: d.risk_color || '#10b981', level: d.alert_level || 'STABLE' }))
        .sort((a, b) => b.risk - a.risk);

    list.innerHTML = '';
    sorted.forEach((p, i) => {
        const item = document.createElement('div');
        item.className = 'triage-item';
        item.onclick   = () => window.open('/patient?pid=' + p.pid, '_blank');
        item.innerHTML = `
            <div class="triage-rank">${i + 1}</div>
            <div class="triage-name">${p.name.split(' ')[0]}</div>
            <div class="triage-risk" style="color:${p.color}">${p.risk.toFixed(0)}%</div>
            <div class="triage-bar" style="width:${p.risk}%;background:${p.color}"></div>
        `;
        list.appendChild(item);
    });
}

// ─── CLINICAL TIMELINE ─────────────────────────────
function addTimelineEvent(pid, evt, prepend) {
    const container = document.getElementById('timeline-events');
    if (!container) return;

    const name = patients[pid]?.name?.split(' ')[0] || pid;
    const div  = document.createElement('div');
    div.className = 'tl-event ' + (evt.type || 'TRIGGER');
    div.innerHTML = `<div class="tl-time">${evt.time}</div><div class="tl-msg">${evt.msg}</div><div class="tl-pid">${name}</div>`;

    if (prepend) container.prepend(div);
    else         container.appendChild(div);

    // Keep max 30
    while (container.children.length > 30) container.lastChild.remove();
}

// ─── ALERT TOAST ───────────────────────────────────
function fireToast(pkt, level = 'CRITICAL') {
    const overlay = document.getElementById('alert-overlay');
    if (!overlay) return;
    const t = document.createElement('div');
    const isCrit = level === 'CRITICAL';
    t.className = 'alert-toast' + (isCrit ? '' : ' warning-toast');
    const icon = isCrit ? '🚨' : '⚠️';
    const titleClass = isCrit ? 'at-title' : 'at-title warn';
    const titleText = isCrit ? 'CRITICAL ALERT' : 'WARNING ALERT';
    t.innerHTML = `<div class="at-icon">${icon}</div><div><div class="${titleClass}">${titleText} — ${pkt.bed || pkt.pid}</div><div class="at-msg">${pkt.name}: ${pkt.risk_score?.toFixed(0)}% · ${(pkt.explanation || [])[0] || 'Elevated sepsis risk'}</div></div>`;
    overlay.prepend(t);
    setTimeout(() => t.remove(), 6500);
}

// ─── EXPLAINABILITY PANEL ──────────────────────────
function renderExplain(pid) {
    const bars = document.getElementById('explain-bars');
    const lbl  = document.getElementById('explain-pid');
    if (!bars || !patients[pid]) return;

    const p = patients[pid];
    lbl.textContent = (p.name || pid) + ' · ' + (p.bed || '');

    const contributions = p.contributions || {};
    const total = Object.values(contributions).reduce((s, v) => s + v, 0) || 1;
    bars.innerHTML = '';

    // Sort descending
    const sorted = Object.entries(contributions).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([label, val]) => {
        const pct = Math.min(Math.round((val / total) * 100), 100);
        const div = document.createElement('div');
        div.className = 'eb';
        div.innerHTML = `<div class="eb-lbl">${label}</div><div class="eb-track"><div class="eb-fill" style="width:0%" data-pct="${pct}"></div></div><div class="eb-pct">${pct}%</div>`;
        bars.appendChild(div);
        // Animate
        requestAnimationFrame(() => { div.querySelector('.eb-fill').style.width = pct + '%'; });
    });
}

// ─── MODAL ─────────────────────────────────────────
function openModal(pid) {
    modalPid = pid;
    selectedPid = pid;
    renderExplain(pid);

    const data = patients[pid];
    if (!data) return;

    document.getElementById('detail-modal').classList.add('open');
    document.getElementById('m-name').textContent = data.name || pid;
    document.getElementById('m-sub').textContent  = (data.bed || '—') + ' · Age ' + (data.age || '—') + ' · Room ' + (data.room || '—');

    updateModal(data);

    // Modal ECG
    const mc = document.getElementById('modal-ecg');
    if (modalEcg) modalEcg.stop();
    modalEcg = new ECGRenderer(mc, { speed: 2.5, lineWidth: 2 });
    if (data.Heart_Rate || data.vitals?.Heart_Rate) modalEcg.setHR(data.Heart_Rate || data.vitals.Heart_Rate);
    modalEcg.setLevel(data.alert_level || 'STABLE');
    modalEcg.start();

    initModalTrend(pid);
}

function updateModal(data) {
    if (!data) return;
    const v = data.vitals || data;

    const vc = (val, lo, hi) => (val < lo || val > hi ? 'danger' : 'ok');

    setModalVital('m-hr',   Math.round(v.Heart_Rate   ?? 0), vc(v.Heart_Rate ?? 75, 60, 100));
    setModalVital('m-bp',   Math.round(v.Blood_Pressure ?? 0), vc(v.Blood_Pressure ?? 110, 90, 120));
    setModalVital('m-spo2', (v.Oxygen_Level ?? 0).toFixed(1), v.Oxygen_Level < 94 ? 'danger' : 'ok');
    setModalVital('m-temp', (v.Temperature ?? 0).toFixed(2), vc(v.Temperature ?? 37, 36.5, 37.5));
    setModalVital('m-rr',   Math.round(v.Resp_Rate ?? 0), vc(v.Resp_Rate ?? 16, 12, 20));
    setModalVital('m-age',  v.Age || data.age || '—', '');
    setModalVital('m-inf',  (v.Infection_Marker ?? 0).toFixed(3), v.Infection_Marker > 0.5 ? 'danger' : 'ok');

    const risk  = data.risk_score ?? 0;
    const color = data.risk_color || '#10b981';
    const rEl   = document.getElementById('m-risk');
    if (rEl) { rEl.textContent = risk.toFixed(0) + '%'; rEl.style.color = color; rEl.className = 'mvc-val'; }
    setText('m-risk-lbl',   'Sepsis Risk Score');
    setText('m-risk-level', data.risk_level || '—');
    setText('m-sirs',  (data.sirs_score ?? '—') + '/4');
    setText('m-qsofa', (data.qsofa_score ?? '—') + '/2');

    const tList = document.getElementById('m-triggers');
    if (tList) {
        const exp = data.explanation || [];
        tList.innerHTML = exp.length === 0
            ? '<span class="trigger-tag ok">All vitals normal</span>'
            : exp.map(e => `<span class="trigger-tag">${e}</span>`).join('');
    }

    if (data.ai_synthesis) setText('m-synthesis', data.ai_synthesis);

    // Trend update
    if (modalChart && data.trend?.length) {
        const trend = data.trend;
        modalChart.data.labels = trend.map((_, i) => {
            const pastSec = (trend.length - 1 - i) * 2;
            return pastSec === 0 ? 'NOW' : `-${pastSec}s`;
        });
        modalChart.data.datasets[0].data = trend;
        modalChart.data.datasets[0].borderColor = color;
        modalChart.data.datasets[0].pointBorderColor = color;
        modalChart.update('none');
    }
}

function setModalVital(id, val, cls) {
    const el = document.getElementById(id);
    if (el) { el.textContent = val; if (cls) el.className = 'mvc-val ' + cls; }
}

function initModalTrend(pid) {
    const canvas = document.getElementById('modal-trend');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const trend = (patients[pid]?.trend && patients[pid].trend.length > 0)
        ? patients[pid].trend
        : Array(20).fill(patients[pid]?.risk_score || 15);
    const color = patients[pid]?.risk_color || '#38bdf8';
    
    if (modalChart) modalChart.destroy();

    const gradient = ctx.createLinearGradient(0, 0, 0, 140);
    gradient.addColorStop(0, color + '55');
    gradient.addColorStop(0.5, color + '22');
    gradient.addColorStop(1, color + '03');

    const labels = trend.map((_, i) => {
        const pastSec = (trend.length - 1 - i) * 2;
        return pastSec === 0 ? 'NOW' : `-${pastSec}s`;
    });

    modalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sepsis Risk (%)',
                data: trend,
                borderColor: color,
                borderWidth: 2.5,
                backgroundColor: gradient,
                fill: true,
                tension: 0.35,
                pointRadius: (context) => (context.dataIndex === context.dataset.data.length - 1 ? 5 : 2),
                pointHoverRadius: 6,
                pointBackgroundColor: (context) => (context.dataIndex === context.dataset.data.length - 1 ? '#ffffff' : color),
                pointBorderColor: color,
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
}

function closeModal() {
    document.getElementById('detail-modal').classList.remove('open');
    if (modalEcg)   { modalEcg.stop();    modalEcg   = null; }
    if (modalChart) { modalChart.destroy(); modalChart = null; }
    modalPid = null;
}

function runModalAssessment() {
    if (!modalPid || !patients[modalPid]) return;
    const btn = document.getElementById('m-assessment-btn');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Running Assessment…'; }
    
    const p = patients[modalPid];
    const payload = {
        Heart_Rate: p.Heart_Rate || p.vitals?.Heart_Rate || 80,
        Oxygen_Level: p.Oxygen_Level || p.vitals?.Oxygen_Level || 98,
        Temperature: p.Temperature || p.vitals?.Temperature || 37.0,
        Blood_Pressure: p.Blood_Pressure || p.vitals?.Blood_Pressure || 120,
        Resp_Rate: p.Resp_Rate || p.vitals?.Resp_Rate || 16,
        Infection_Marker: p.Infection_Marker || p.vitals?.Infection_Marker || 0.5,
        Age: p.Age || p.vitals?.Age || 65,
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
    .then(res => res.json())
    .then(data => {
        if (btn) { btn.disabled = false; btn.textContent = '⚡ Run Risk Assessment'; }
        patients[modalPid] = { ...patients[modalPid], ...data };
        updateModal(patients[modalPid]);
    })
    .catch(err => {
        if (btn) { btn.disabled = false; btn.textContent = '⚡ Run Risk Assessment'; }
        console.error('[Assessment Error]', err);
    });
}

function requestAI() {
    if (!modalPid) return;
    const btn = document.getElementById('m-ai-btn');
    btn.textContent = '⚡ Generating…'; btn.disabled = true;
    setText('m-synthesis', 'AI synthesis generating…');
    socket.emit('request_ai_synthesis', { pid: modalPid });
}

function openPatientPage() {
    if (!modalPid) return;
    window.open('/patient?pid=' + modalPid, '_blank');
}

function typewriter(el, text) {
    if (!el) return;
    el.textContent = '';
    let i = 0;
    const t = setInterval(() => { if (i < text.length) el.textContent += text[i++]; else clearInterval(t); }, 13);
}

// Keyboard / backdrop close
document.getElementById('detail-modal').addEventListener('click', function(e) { if (e.target === this) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ─── ECG RENDERER ─────────────────────────────────
class ECGRenderer {
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');
        this.speed  = opts.speed     || 2;
        this.lw     = opts.lineWidth || 1.5;
        this.hr     = 75;
        this.color  = '#10b981';
        this.phase  = Math.random() * Math.PI * 2;
        this.buf    = [];
        this.on     = false;
        this._raf   = null;
        this._resize();
        if (canvas.parentElement)
            new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    }

    _resize() {
        const p = this.canvas.parentElement;
        if (!p) return;
        this.canvas.width  = p.clientWidth;
        this.canvas.height = p.clientHeight;
        this.W = this.canvas.width;
        this.H = this.canvas.height;
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

        const midY = H * 0.5;
        const amp  = H * 0.36;
        const inc  = (this.hr / 60) * (Math.PI * 2) / 60 * speed;

        for (let s = 0; s < speed; s++) {
            this.phase += inc / speed;
            const y = midY - this._sample(this.phase) * amp + (Math.random() - 0.5) * 0.4;
            this.buf.push(y);
        }
        if (this.buf.length > W + speed) this.buf.splice(0, this.buf.length - W);

        ctx.clearRect(0, 0, W, H);

        // grid
        ctx.strokeStyle = 'rgba(56,189,248,0.04)';
        ctx.lineWidth   = 0.5;
        for (let gy = 0.25; gy < 1; gy += 0.25) {
            ctx.beginPath(); ctx.moveTo(0, H * gy); ctx.lineTo(W, H * gy); ctx.stroke();
        }

        // trace
        ctx.save();
        ctx.shadowColor = this.color; ctx.shadowBlur = 7;
        ctx.strokeStyle = this.color; ctx.lineWidth  = this.lw;
        ctx.lineJoin    = 'round';
        ctx.beginPath();
        const start = Math.max(0, this.buf.length - W);
        for (let x = 0; x < Math.min(this.buf.length, W); x++) {
            const y = this.buf[start + x];
            x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();

        this._raf = requestAnimationFrame(() => this._render());
    }
}

// ─── PARTICLE SYSTEM ─────────────────────────────
(function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const particles = [];
    const N = 55;
    function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
    window.addEventListener('resize', resize);
    resize();
    for (let i = 0; i < N; i++) {
        particles.push({ x: Math.random() * canvas.width, y: Math.random() * canvas.height,
            r: Math.random() * 1.2 + 0.3, vx: (Math.random() - 0.5) * 0.18, vy: (Math.random() - 0.5) * 0.18,
            a: Math.random() * 0.5 + 0.1 });
    }
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;  if (p.x > canvas.width)  p.x = 0;
            if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(56,189,248,' + p.a + ')'; ctx.fill();
        });
        for (let i = 0; i < N; i++) {
            for (let j = i + 1; j < N; j++) {
                const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
                const d  = Math.sqrt(dx * dx + dy * dy);
                if (d < 90) {
                    ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y); ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = 'rgba(56,189,248,' + (0.07 * (1 - d / 90)) + ')';
                    ctx.lineWidth = 0.5; ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();

// ─── ICU FLOOR MAP ────────────────────────────────
const floorMap = (function() {
    const canvas = document.getElementById('floor-canvas');
    if (!canvas) return { update: () => {} };
    const ctx = canvas.getContext('2d');
    let pulse = 0;
    const beds = [
        ['P001', 0.04, 0.05, 0.28, 0.42], ['P002', 0.36, 0.05, 0.28, 0.42],
        ['P003', 0.68, 0.05, 0.28, 0.42], ['P004', 0.04, 0.55, 0.28, 0.42],
        ['P005', 0.36, 0.55, 0.28, 0.42], ['P006', 0.68, 0.55, 0.28, 0.42],
    ];
    function draw() {
        const W = canvas.width  = canvas.parentElement.clientWidth;
        const H = canvas.height = canvas.parentElement.clientHeight;
        ctx.clearRect(0, 0, W, H);
        pulse = (pulse + 0.05) % (Math.PI * 2);
        beds.forEach(([pid, bx, by, bw, bh]) => {
            const p = patients[pid], level = p?.alert_level || 'STABLE', risk = p?.risk_score || 0, color = p?.risk_color || '#10b981';
            const x = bx * W, y = by * H, w = bw * W, h = bh * H;
            ctx.shadowBlur  = level === 'CRITICAL' ? 8 + Math.sin(pulse) * 5 : 0;
            ctx.shadowColor = color;
            ctx.fillStyle   = level === 'CRITICAL' ? 'rgba(239,68,68,0.12)' : level === 'WARNING' ? 'rgba(245,158,11,0.1)' : 'rgba(16,185,129,0.06)';
            ctx.strokeStyle = color; ctx.lineWidth = level === 'CRITICAL' ? 1.5 : 1;
            ctx.beginPath(); ctx.roundRect(x, y, w, h, 4); ctx.fill(); ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.fillStyle = color + '44'; ctx.fillRect(x, y + h - 3, w, 3);
            ctx.fillStyle = color;        ctx.fillRect(x, y + h - 3, w * (risk / 100), 3);
            ctx.fillStyle = '#94a3b8'; ctx.font = 'bold ' + Math.round(W * 0.036) + 'px "JetBrains Mono",monospace'; ctx.textAlign = 'center';
            ctx.fillText(p?.bed || pid, x + w / 2, y + h * 0.38);
            ctx.fillStyle = color; ctx.font = 'bold ' + Math.round(W * 0.044) + 'px "JetBrains Mono",monospace';
            ctx.fillText(risk.toFixed(0) + '%', x + w / 2, y + h * 0.72);
            if (level === 'CRITICAL') {
                ctx.beginPath(); ctx.arc(x + w - 7, y + 7, 3 + Math.sin(pulse * 2) * 1.5, 0, Math.PI * 2);
                ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 8; ctx.fill(); ctx.shadowBlur = 0;
            }
        });
        requestAnimationFrame(draw);
    }
    draw();
    return { update: () => {} };
})();

// ─── AI COPILOT ───────────────────────────────────
function sendCopilot() {
    const input = document.getElementById('cop-input');
    const q = input.value.trim();
    if (!q) return;
    appendCopilotMsg(q, 'user');
    input.value = '';
    const msgs = document.getElementById('copilot-msgs');
    const think = document.createElement('div');
    think.className = 'cop-msg ai'; think.id = 'cop-thinking';
    think.innerHTML = '<div class="cop-thinking"><span></span><span></span><span></span></div>';
    msgs.appendChild(think); msgs.scrollTop = msgs.scrollHeight;
    socket.emit('copilot_query', { question: q, pid: modalPid || selectedPid || null });
}

function appendCopilotMsg(text, role) {
    const msgs = document.getElementById('copilot-msgs');
    const div  = document.createElement('div');
    div.className = 'cop-msg ' + role; div.textContent = text;
    msgs.appendChild(div); msgs.scrollTop = msgs.scrollHeight;
}

function removeCopilotThinking() {
    const el = document.getElementById('cop-thinking');
    if (el) el.remove();
}

document.getElementById('cop-input').addEventListener('keydown', e => { if (e.key === 'Enter') sendCopilot(); });

console.log('[SepsisGuard v3.0] ICU Intelligence Ecosystem — All systems online.');
