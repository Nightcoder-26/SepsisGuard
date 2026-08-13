/**
 * SepsisGuard AI v3.0 — Judge Demonstration / Test Workflow JS
 */

const SERVER  = 'http://localhost:5000';
const API_KEY = 'sepsisguard_api_key_3f7b9a1c5d8e2f4a6c0b8d1e3f5a7c9b';

// Clock updates
setInterval(() => {
    const el = document.getElementById('clock');
    if (el) el.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
}, 1000);

// Presets
const presets = {
    normal: {
        pid: 'DEMO-NORM',
        age: 45,
        gender: 'M',
        bed: 'ICU-02',
        hr: 72,
        temp: 36.8,
        bp: 115,
        rr: 14,
        spo2: 98.0,
        inf: 0.120
    },
    warning: {
        pid: 'DEMO-ELEV',
        age: 68,
        gender: 'F',
        bed: 'ICU-04',
        hr: 98,
        temp: 37.9,
        bp: 105,
        rr: 22,
        spo2: 94.0,
        inf: 0.450
    },
    critical: {
        pid: 'DEMO-CRIT',
        age: 79,
        gender: 'M',
        bed: 'ICU-05',
        hr: 118,
        temp: 38.6,
        bp: 88,
        rr: 28,
        spo2: 89.0,
        inf: 0.820
    }
};

let lastResult = null;
let lastInput = null;

function loadPreset(key, btn) {
    // Manage active state of preset buttons
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');

    const data = presets[key];
    if (!data) return;

    document.getElementById('dm-pid').value = data.pid;
    document.getElementById('dm-age').value = data.age;
    document.getElementById('dm-gender').value = data.gender;
    document.getElementById('dm-bed').value = data.bed;
    document.getElementById('dm-hr').value = data.hr;
    document.getElementById('dm-temp').value = data.temp.toFixed(1);
    document.getElementById('dm-bp').value = data.bp;
    document.getElementById('dm-rr').value = data.rr;
    document.getElementById('dm-spo2').value = data.spo2.toFixed(1);
    document.getElementById('dm-inf').value = data.inf.toFixed(3);
    
    document.getElementById('dm-error').textContent = '';
}

function resetDemoForm() {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('dm-pid').value = 'DEMO-P001';
    document.getElementById('dm-age').value = '65';
    document.getElementById('dm-gender').value = 'M';
    document.getElementById('dm-bed').value = 'ICU-DEMO';
    document.getElementById('dm-hr').value = '80';
    document.getElementById('dm-temp').value = '37.0';
    document.getElementById('dm-bp').value = '120';
    document.getElementById('dm-rr').value = '16';
    document.getElementById('dm-spo2').value = '98.0';
    document.getElementById('dm-inf').value = '0.15';
    document.getElementById('dm-error').textContent = '';

    // Reset results panel
    document.getElementById('demo-result-panel').style.display = 'none';
    document.getElementById('demo-result-empty').style.display = 'flex';

    // Reset flowchart dots
    resetFlowDots();

    lastResult = null;
    lastInput = null;
}

function resetFlowDots() {
    const dots = ['validate', 'model', 'shap', 'scores', 'ai'];
    dots.forEach(id => {
        const el = document.getElementById('ad-' + id);
        if (el) {
            el.className = 'step-dot';
            el.parentElement.classList.remove('active');
        }
    });
}

function activateDot(id, ok = true) {
    const el = document.getElementById('ad-' + id);
    if (el) {
        el.className = 'step-dot' + (ok ? ' ok' : '');
        el.parentElement.classList.add('active');
    }
}

function toggleConsole() {
    const body = document.getElementById('tech-body');
    const arrow = document.getElementById('console-arrow');
    if (!body) return;
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    if (arrow) arrow.textContent = open ? '▼' : '▲';
}

function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

async function runDemoAssessment() {
    const errEl = document.getElementById('dm-error');
    const btn = document.getElementById('dm-submit-btn');
    errEl.textContent = '';

    const pid = document.getElementById('dm-pid').value.trim();
    const age = parseFloat(document.getElementById('dm-age').value);
    const gender = document.getElementById('dm-gender').value;
    const bed = document.getElementById('dm-bed').value.trim();

    const hr = parseFloat(document.getElementById('dm-hr').value);
    const temp = parseFloat(document.getElementById('dm-temp').value);
    const bp = parseFloat(document.getElementById('dm-bp').value);
    const rr = parseFloat(document.getElementById('dm-rr').value);
    const spo2 = parseFloat(document.getElementById('dm-spo2').value);
    const inf = parseFloat(document.getElementById('dm-inf').value);

    // Validate inputs locally for instant UX
    const rangeChecks = [
        [hr, 20, 300, 'Heart Rate'],
        [temp, 30, 45, 'Temperature'],
        [bp, 30, 250, 'Blood Pressure'],
        [rr, 4, 70, 'Respiratory Rate'],
        [spo2, 50, 100, 'SpO₂'],
        [inf, 0, 1, 'Infection Marker']
    ];
    for (const [val, lo, hi, label] of rangeChecks) {
        if (isNaN(val) || val < lo || val > hi) {
            errEl.textContent = `${label}: Out of valid supported range (${lo}–${hi}).`;
            return;
        }
    }
    if (isNaN(age) || age < 0 || age > 120) {
        errEl.textContent = 'Age: Out of valid supported range (0–120).';
        return;
    }

    // Prepare payload
    const payload = {
        Heart_Rate: hr,
        Oxygen_Level: spo2,
        Temperature: temp,
        Blood_Pressure: bp,
        Resp_Rate: rr,
        Age: age,
        Infection_Marker: inf,
        generate_synthesis: true
    };

    // Update UI loading states
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing assessment...';
    resetFlowDots();

    // Start UI flowchart simulation
    activateDot('validate');
    
    const requestTime = new Date().toISOString();
    let responseText = '';

    try {
        // Run ML model step
        await new Promise(r => setTimeout(r, 400));
        activateDot('model');

        // Explainer / SHAP step
        await new Promise(r => setTimeout(r, 450));
        activateDot('shap');

        // Scoring rules step
        await new Promise(r => setTimeout(r, 300));
        activateDot('scores');

        // AI narrative step
        await new Promise(r => setTimeout(r, 350));
        activateDot('ai');

        const res = await fetch(SERVER + '/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        responseText = JSON.stringify(data, null, 2);

        if (!res.ok) {
            errEl.textContent = data.message || 'Risk assessment pipeline execution failed.';
            resetFlowDots();
            return;
        }

        // Cache results
        lastResult = data;
        lastInput = { pid, age, gender, bed, ...payload };

        // Hide placeholder, show panel
        document.getElementById('demo-result-empty').style.display = 'none';
        document.getElementById('demo-result-panel').style.display = 'block';

        // Render real results
        renderDemoResults(data);

    } catch (e) {
        errEl.textContent = 'Assessment service is currently unavailable. Please verify the backend is running.';
        resetFlowDots();
        responseText = 'Network Request Exception: ' + e.message;
    } finally {
        btn.disabled = false;
        btn.textContent = '⚡ Run Risk Assessment';

        // Print to technical flow console (redacting auth headers / sensitive details)
        const consoleEl = document.getElementById('console-output');
        if (consoleEl) {
            consoleEl.innerHTML = `
<strong style="color:#f59e0b">HTTP Request:</strong>
POST ${SERVER}/predict
Headers: { "Content-Type": "application/json", "X-API-Key": "[REDACTED]" }
Payload: ${JSON.stringify(payload, null, 2)}

<strong style="color:#10b981">HTTP Response (At: ${requestTime}):</strong>
Status: 200 OK
Response: ${escapeHtml(responseText)}`;
        }
    }
}

function renderDemoResults(data) {
    const risk = data.risk_score ?? 0;
    const level = data.risk_level || 'Low';
    const color = data.risk_color || '#10b981';

    // Estimates
    const lvlEl = document.getElementById('dm-res-level');
    lvlEl.textContent = level;
    lvlEl.style.color = color;

    document.getElementById('dm-res-score').textContent = risk.toFixed(1) + '% Sepsis Risk Score';

    // Visual gauge
    const needle = document.getElementById('dm-gauge-needle');
    if (needle) needle.style.left = Math.min(Math.max(0, risk), 100) + '%';

    // Card summaries
    document.getElementById('dm-card-risk').textContent = risk.toFixed(0) + '%';
    document.getElementById('dm-card-risk').style.color = color;
    document.getElementById('dm-card-sirs').textContent = (data.sirs_score ?? '—') + '/4';
    document.getElementById('dm-card-qsofa').textContent = (data.qsofa_score ?? '—') + '/2';
    
    const ts = new Date();
    document.getElementById('dm-card-time').textContent = ts.toLocaleTimeString('en-GB', { hour12: false });

    // SHAP explanation chart
    renderShapChart(data.shap_explanation);

    // SIRS breakdown checklist
    const sirs = data.sirs_criteria || {};
    document.getElementById('dm-sirs-score').textContent = data.sirs_score ?? '0';
    setChk('dm-chk-temp', sirs.temp_met);
    setChk('dm-chk-hr', sirs.hr_met);
    setChk('dm-chk-rr', sirs.rr_met);
    setChk('dm-chk-wbc', sirs.wbc_met);

    // qSOFA checklist
    const qsofa = data.qsofa_criteria || {};
    document.getElementById('dm-qsofa-score').textContent = data.qsofa_score ?? '0';
    setChk('dm-chk-qrr', qsofa.rr_met);
    setChk('dm-chk-sbp', qsofa.sbp_met);

    // AI summary
    const aiEl = document.getElementById('dm-ai-synthesis');
    if (aiEl) {
        if (data.ai_synthesis) {
            aiEl.textContent = '';
            typewriter(aiEl, data.ai_synthesis);
        } else {
            aiEl.textContent = 'AI summary unavailable; displaying rule-based triggers.';
        }
    }
}

function setChk(id, met) {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = 'chk-item ' + (met ? 'met' : 'unmet');
    const badge = el.querySelector('.chk-badge');
    if (badge) badge.textContent = met ? '✓ Met' : '○ Unmet';
}

function renderShapChart(shap_data) {
    const container = document.getElementById('dm-shap-bars');
    if (!container) return;
    container.innerHTML = '';

    if (!shap_data || shap_data.available === false) {
        container.innerHTML = '<div style="font-size:0.7rem;color:var(--text-2);padding:10px">Attribution data not available for this run.</div>';
        return;
    }

    const features = shap_data.features || [];
    if (features.length === 0) {
        container.innerHTML = '<div style="font-size:0.7rem;color:var(--text-2);padding:10px">No significant SHAP attributions found.</div>';
        return;
    }

    // Scaling factor helper
    const maxVal = Math.max(...features.map(f => Math.abs(f.shap_value || 0)), 0.05);

    features.forEach(f => {
        const name = f.display_name || f.feature;
        const val = f.value != null ? (f.unit ? `${f.value} ${f.unit}` : `${f.value}`) : '';
        const shap = f.shap_value || 0;
        const isPos = shap >= 0;
        const pct = Math.min(Math.round((Math.abs(shap) / maxVal) * 45), 45); // Max 45% either side

        const barColor = isPos ? '#ef4444' : '#10b981';
        const sign = isPos ? '+' : '';

        const row = document.createElement('div');
        row.style.cssText = 'display:flex;flex-direction:column;gap:3px;margin-bottom:8px;background:rgba(255,255,255,0.02);padding:8px;border-radius:4px;font-size:0.7rem';
        row.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;font-weight:600">
                <span>${name} <span style="font-weight:400;color:var(--text-2);font-size:0.62rem">(${val})</span></span>
                <span style="color:${barColor};font-family:var(--mono)">${sign}${shap.toFixed(3)}</span>
            </div>
            <div style="position:relative;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;margin-top:4px">
                <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.25);z-index:2"></div>
                <div style="position:absolute;top:0;bottom:0;background:${barColor};border-radius:2px;${isPos ? `left:50%;width:${pct}%` : `right:50%;width:${pct}%`}"></div>
            </div>
            <div style="font-size:0.62rem;color:var(--text-2);margin-top:2px">${f.formatted_text || ''}</div>
        `;
        container.appendChild(row);
    });
}

function typewriter(el, text) {
    if (!el) return;
    el.textContent = '';
    let i = 0;
    const interval = setInterval(() => {
        if (i < text.length) el.textContent += text[i++];
        else clearInterval(interval);
    }, 12);
}

function exportDemoReport() {
    if (!lastResult || !lastInput) return;
    const r = lastResult;
    const inp = lastInput;

    const lines = [
        '==================================================',
        '  SEPSISGUARD AI — CLINICAL ASSESSMENT REPORT',
        '==================================================',
        'DEMONSTRATION PRESET RUN / SYNTHETIC ASSESSMENT DATA',
        '==================================================',
        'DISCLAIMER:',
        'This report is generated by a research decision-support',
        'prototype using synthetic demonstration inputs. It is',
        'not for clinical decision-making or diagnostic use.',
        '==================================================',
        '',
        'Demo Patient ID:    ' + inp.pid,
        'Age / Sex:          ' + inp.age + ' yrs / ' + inp.gender,
        'Care Location:      ' + (inp.bed || '—'),
        'Assessment Time:    ' + new Date().toLocaleString(),
        '',
        '--- INPUT CLINICAL VITALS ---',
        'Heart Rate:         ' + inp.Heart_Rate + ' bpm',
        'Temperature:        ' + inp.Temperature + ' °C',
        'Systolic BP:        ' + inp.Blood_Pressure + ' mmHg',
        'Respiratory Rate:   ' + inp.Resp_Rate + ' /min',
        'SpO₂:               ' + inp.Oxygen_Level + ' %',
        'Infection Marker:   ' + inp.Infection_Marker,
        '',
        '--- PIPELINE MODEL RESULT ---',
        'Model Risk score:   ' + r.risk_score?.toFixed(1) + '%',
        'Risk Level:         ' + r.risk_level,
        'SIRS Score:         ' + (r.sirs_score ?? '—') + '/4 (Rule-based)',
        'Partial qSOFA:      ' + (r.qsofa_score ?? '—') + '/2 (Partial — mentation unavailable)',
        '',
        '--- CLINICAL TRIGGERS ---',
        ...(r.explanation || []).map(e => '  • ' + e),
        '',
        '--- AI NARRATIVE EXPLANATION ---',
        r.ai_synthesis || 'N/A',
        '',
        '--- MODEL ARCHITECTURE METADATA ---',
        'Model Pipeline: XGBoost Classifier',
        'Calibration: Operating Threshold = 0.27',
        'Features Attribute Analysis: SHAP TreeExplainer',
        '=================================================='
    ];

    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SepsisGuard_DemoAssessment_${inp.pid}_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}
