/** Stellar Predictor — Pattern Analysis Frontend */

let currentTaskId = null;
let pollInterval = null;
let pollCount = 0;
const MAX_POLLS = 120;

document.addEventListener('DOMContentLoaded', () => {
    loadSystems();
});

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiGet(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} GET ${path}`);
    return resp.json();
}

async function apiPost(path, body) {
    const resp = await fetch(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} POST ${path}`);
    return resp.json();
}

function showError(context, err) {
    console.error(context, err);
    const el = document.getElementById('error-banner');
    if (el) {
        el.textContent = `${context}: ${err.message || err}`;
        el.style.display = 'block';
        setTimeout(() => { el.style.display = 'none'; }, 8000);
    }
}

// ---------------------------------------------------------------------------
// System loading
// ---------------------------------------------------------------------------
async function loadSystems() {
    try {
        const systems = await apiGet('/api/systems');
        const select = document.getElementById('system-select');
        select.innerHTML = systems.map(s =>
            `<option value="${s.name}">${s.display_name} (${s.planet_count} planets)</option>`
        ).join('');
        if (systems.length > 0) {
            select.value = systems[0].name;
            loadPlanets(systems[0].name);
        }
    } catch (e) {
        showError('Failed to load systems', e);
    }
}

document.getElementById('system-select').addEventListener('change', (e) => {
    loadPlanets(e.target.value);
});

async function loadPlanets(systemName) {
    try {
        const data = await apiGet(`/api/systems/${systemName}/planets`);
        if (data.error) { showError('Load planets', data.error); return; }
        const planetList = document.getElementById('planet-list');
        planetList.innerHTML = data.planets.map(p => {
            const mass = p.mass_earth != null
                ? ` <span style="color:#8b949e">(${p.mass_earth.toFixed(1)} M&#x2295;)</span>`
                : '';
            return `<span class="planet-chip">${p.name}<span class="a-label">${p.semi_major_axis_au} AU${mass}</span></span>`;
        }).join('');
    } catch (e) {
        showError('Failed to load planets', e);
    }
}

// ---------------------------------------------------------------------------
// Analysis lifecycle
// ---------------------------------------------------------------------------
async function startAnalysis() {
    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    document.getElementById('results-panel').style.display = 'none';
    document.getElementById('gaps-list').innerHTML = '';
    document.getElementById('distribution-plot').innerHTML = '<p class="placeholder">Running analysis... / 分析中...</p>';
    document.getElementById('report-container').innerHTML = '<p class="placeholder" style="padding:2rem;text-align:center">Waiting... / 等待中...</p>';

    const system = document.getElementById('system-select').value;
    try {
        const data = await apiPost('/api/analyze', {system, mode: 'pattern_analysis'});
        currentTaskId = data.task_id;
        pollCount = 0;

        document.getElementById('progress-panel').style.display = 'block';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = 'Task submitted... / 已提交...';
        btn.textContent = 'Running... / 运行中...';
        startPolling();
    } catch (e) {
        showError('Analysis start failed', e);
        btn.disabled = false;
        btn.textContent = 'Analyze / 开始分析';
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(checkTaskStatus, 800);
}

async function checkTaskStatus() {
    if (!currentTaskId) return;
    pollCount++;
    if (pollCount > MAX_POLLS) {
        clearInterval(pollInterval); pollInterval = null;
        document.getElementById('progress-text').textContent = 'Timed out. Please try again. / 超时，请重试';
        document.getElementById('analyze-btn').disabled = false;
        document.getElementById('analyze-btn').textContent = 'Analyze / 开始分析';
        return;
    }

    try {
        const data = await apiGet(`/api/analyze/${currentTaskId}`);
        const pct = Math.round((data.progress || 0) * 100);
        document.getElementById('progress-fill').style.width = pct + '%';

        const stageNames = {
            loading_system: 'Loading system / 加载系统中...',
            pattern_analysis: 'Pattern analysis / 规律分析中...',
            generating_visualizations: 'Generating chart / 生成图表...',
            generating_report: 'Generating report / 生成报告中...',
            formatting_results: 'Formatting / 格式化结果...',
            complete: 'Complete / 完成',
            error: 'Error / 错误',
        };
        document.getElementById('progress-text').textContent =
            `${stageNames[data.stage] || data.stage || 'Processing...'} (${pct}%)`;

        if (data.status === 'complete') {
            clearInterval(pollInterval); pollInterval = null;

            if (data.result) {
                try { loadResults(data.result); } catch (e) { showError('Display results', e); }
                loadDistributionPlot();
            } else {
                document.getElementById('progress-text').textContent = 'Complete, no result. / 完成，无结果';
            }

            document.getElementById('analyze-btn').disabled = false;
            document.getElementById('analyze-btn').textContent = 'Analyze / 开始分析';
            document.getElementById('progress-panel').style.display = 'none';

        } else if (data.status === 'failed') {
            clearInterval(pollInterval); pollInterval = null;
            showError('Analysis failed', data.error || 'Unknown error');
            document.getElementById('analyze-btn').disabled = false;
            document.getElementById('analyze-btn').textContent = 'Analyze / 开始分析';
        }
    } catch (e) {
        if (pollCount > 5) showError('Polling error', e);
    }
}

// ---------------------------------------------------------------------------
// Gap cards
// ---------------------------------------------------------------------------
function loadResults(result) {
    if (!result) return;

    const tbSummary = document.getElementById('tb-summary');
    if (result.tb_fit) {
        const tb = result.tb_fit;
        tbSummary.innerHTML =
            `Titius-Bode: a<sub>n</sub> = ${tb.alpha.toFixed(4)} &times; ${tb.beta.toFixed(4)}<sup>n</sup> | R&sup2; = ${tb.r_squared.toFixed(4)}`;
    } else {
        tbSummary.textContent = 'Titius-Bode: insufficient planets for fit';
    }

    document.querySelector('.analysis-status').textContent =
        `Analyzed in ${result.execution_time_s}s / ${result.num_known_planets} planets`;

    const gapsList = document.getElementById('gaps-list');
    const gaps = result.gaps || [];
    if (gaps.length === 0) {
        gapsList.innerHTML = '<p style="color:#8b949e;padding:1rem">No significant gaps detected.</p>';
        return;
    }

    gapsList.innerHTML = gaps.map(g => {
        const scoreClass = g.combined_score >= 0.5 ? 'gap-score-high'
            : g.combined_score >= 0.3 ? 'gap-score-mid' : 'gap-score-weak';
        const label = g.combined_score >= 0.5 ? 'Strong'
            : g.combined_score >= 0.3 ? 'Possible' : 'Weak';

        return `<div class="gap-card">
            <div class="gap-header">
                <span class="gap-rank">#${g.index} ${g.inner_planet} &rarr; ${g.outer_planet}</span>
                <span class="gap-score-badge ${scoreClass}">${label} (${g.combined_score.toFixed(2)})</span>
            </div>
            <div class="gap-body">
                <div>Pred. orbit / 预测轨道: <span class="pred-a">${g.predicted_a_au.toFixed(2)} AU</span></div>
                <div class="pred-period">Period / 周期: ${g.predicted_period_years.toFixed(1)} yr</div>
                <div style="font-size:0.75rem;color:#8b949e">Mass / 质量: ${g.estimated_mass_min.toFixed(2)} &ndash; ${Number(g.estimated_mass_max).toFixed(0)} M&#x2295;</div>
            </div>
            <div class="gap-scores">
                <span>TB ${g.titius_bode_score.toFixed(2)}</span>
                <span>Stab ${g.stability_score.toFixed(2)}</span>
                <span>${g.method}</span>
            </div>
        </div>`;
    }).join('');

    // Render prediction report
    loadReport(result);
}

// ---------------------------------------------------------------------------
// Distribution plot
// ---------------------------------------------------------------------------
async function loadDistributionPlot() {
    const el = document.getElementById('distribution-plot');
    if (!el) return;

    if (typeof Plotly === 'undefined') {
        el.innerHTML = '<p class="placeholder" style="color:#d29922">Plotly not loaded / CDN 未加载 — chart unavailable</p>';
        return;
    }

    try {
        const data = await apiGet(`/api/viz/distribution/${currentTaskId}`);
        if (!data || data.error) {
            el.innerHTML = `<p class="placeholder" style="color:#f85149">${(data && data.error) || 'No data'}</p>`;
            return;
        }
        if (!data.data || !data.layout) {
            el.innerHTML = '<p class="placeholder" style="color:#f85149">Invalid plot data structure</p>';
            return;
        }
        Plotly.newPlot('distribution-plot', data.data, data.layout, {
            responsive: true, displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        });
    } catch (e) {
        el.innerHTML = `<p class="placeholder" style="color:#f85149">Chart render failed: ${e.message || e}</p>`;
        showError('Distribution plot', e);
    }
}

// ---------------------------------------------------------------------------
// Bilingual prediction report
// ---------------------------------------------------------------------------
function loadReport(result) {
    const container = document.getElementById('report-container');
    const report = result.report;
    if (!report || !report.predicted_bodies || report.predicted_bodies.length === 0) {
        container.innerHTML = '<p style="color:#8b949e;padding:2rem;text-align:center">No predictions to report. / 无预测结果。</p>';
        return;
    }

    let html = '';

    // System reference
    const ref = report.system_reference;
    if (ref) {
        html += `<div class="report-card system-ref">
            <div class="report-header">
                <span class="report-title">System Reference / 系统参考</span>
            </div>
            <div class="report-body">
                <div class="param-row"><span class="param-label">Host star / 主星</span><span class="param-value">M=${ref.star.mass} M_sun, R=${ref.star.radius} R_sun, T_eff=${ref.star.teff} K</span></div>
                <div class="param-row"><span class="param-label">Known planets / 已知行星</span><span class="param-value">${ref.num_known_planets}</span></div>`;
        if (ref.tb_fit) {
            html += `<div class="param-row"><span class="param-label">TB fit / TB拟合</span><span class="param-value">&alpha;=${ref.tb_fit.alpha}, &beta;=${ref.tb_fit.beta}, R&sup2;=${ref.tb_fit.r_squared}</span></div>`;
        }
        html += `<div class="param-row"><span class="param-label">Stability regions / 稳定区</span><span class="param-value">${ref.stability_regions}</span></div>
                <div class="param-row"><span class="param-label">Analysis time / 分析耗时</span><span class="param-value">${ref.execution_time_s}s</span></div>`;
        if (ref.warnings && ref.warnings.length > 0) {
            html += `<div class="param-row"><span class="param-label" style="color:#d29922">Warnings / 警告</span><span class="param-value" style="color:#d29922">${ref.warnings.join('; ')}</span></div>`;
        }
        html += `</div></div>`;
    }

    // Each predicted body
    report.predicted_bodies.forEach(body => {
        html += `<div class="report-card">
            <div class="report-header">
                <span class="report-title">Predicted Body #${body.index} / 预测天体 #${body.index}</span>
                <span class="report-score">Score / 评分: ${body.combined_score.toFixed(2)}</span>
            </div>
            <div class="report-body">`;

        body.params.forEach(p => {
            html += `<div class="param-row">
                <span class="param-label">${p.label_zh} / ${p.label_en}</span>
                <span class="param-value">${p.value} ${p.unit}</span>
            </div>`;
        });

        html += `</div>
            <div class="report-reference">
                Inner / 内邻: ${body.inner_planet.name} (${body.inner_planet.a_au} AU)
                | Outer / 外邻: ${body.outer_planet.name} (${body.outer_planet.a_au} AU)
                | Method: ${body.method}
            </div>
        </div>`;
    });

    container.innerHTML = html;
}
