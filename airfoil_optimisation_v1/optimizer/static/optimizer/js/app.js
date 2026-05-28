const $ = (id) => document.getElementById(id);

const metricsEl = $('metrics');
const constraintsEl = $('constraints');
const pipelineEl = $('pipeline');
const experimentEl = $('experiment');
const xaiEl = $('xai');
const decisionLogicEl = $('decisionLogic');
const modelStatus = $('modelStatus');
const backendStatus = $('backendStatus');

const initialPathEl = $('initialAirfoil');
const optimizedFillEl = $('optimizedAirfoilFill');
const optimizedStrokeEl = $('optimizedAirfoilStroke');
const flowMarkerEl = $('flowMarker');

const chartOx = 120;
const chartOy = 160;
const chartSx = 740;
const chartSy = 720;

let previousMetrics = {
  cl: 0.923,
  cd: 0.0112,
  cl_cd: 82.4,
  cm: -0.0520,
  tc: 0.109
};

function parseWeights(value) {
  return value
    .split(',')
    .map((x) => Number(x.trim().replace(',', '.')))
    .filter((x) => !Number.isNaN(x));
}

function parseNumberInput(id, label) {
  const el = $(id);

  if (!el) {
    throw new Error(`${label} input was not found.`);
  }

  const raw = String(el.value).trim().replace(',', '.');
  const value = Number(raw);

  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a valid number.`);
  }

  return value;
}

function setBackendStatus(type, text) {
  if (!backendStatus) return;
  backendStatus.className = `status-pill status-${type}`;
  backendStatus.innerHTML = `<span></span>${text}`;
}

function setButtonLoading(isLoading) {
  const btn = $('optimizeBtn');

  if (!btn) {
    return;
  }

  const icon = btn.querySelector('.btn-icon');
  const label = btn.querySelector('.btn-label');

  btn.disabled = isLoading;

  if (!icon || !label) {
    btn.textContent = isLoading ? 'Analyzing...' : '⚡ Optimize & Explain';
    return;
  }

  if (isLoading) {
    icon.textContent = '';
    icon.classList.add('loading');
    label.textContent = 'Analyzing...';
  } else {
    icon.classList.remove('loading');
    icon.textContent = '⚡';
    label.textContent = 'Optimize & Explain';
  }
}

function formatMetric(key, value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return '—';

  if (key === 'tc') return `${(value * 100).toFixed(1)}%`;
  if (key === 'cd') return Number(value).toFixed(4);
  if (key === 'cm') return Number(value).toFixed(4);
  if (key === 'cl_cd') return Number(value).toFixed(1);

  return Number(value).toFixed(4);
}

function formatDelta(current, previous, inverse = false) {
  if (!previous || previous === 0) return '—';

  const raw = ((current - previous) / Math.abs(previous)) * 100;
  const score = inverse ? -raw : raw;

  const sign = score > 0 ? '+' : '';
  return `${sign}${score.toFixed(1)}%`;
}

function airfoilToPath(points) {
  if (!points || !points.length) return '';

  let d = `M ${chartOx + points[0][0] * chartSx} ${chartOy - points[0][1] * chartSy}`;

  for (let i = 1; i < points.length; i++) {
    d += ` L ${chartOx + points[i][0] * chartSx} ${chartOy - points[i][1] * chartSy}`;
  }

  d += ' Z';
  return d;
}

function buildDefaultGeometry() {
  const xs = Array.from({ length: 101 }, (_, i) => i / 100);

  const upper = xs.map((x) => [
    x,
    0.036 * Math.sin(Math.PI * x) * (1 - 0.10 * x)
  ]);

  const lower = xs.map((x) => [
    x,
    -0.031 * Math.sin(Math.PI * x) * (1 - 0.05 * x)
  ]);

  return { upper, lower };
}

function renderInitialOnly() {
  if (!initialPathEl) return;

  const geom = buildDefaultGeometry();
  const pts = [
    ...geom.upper,
    ...[...geom.lower].reverse()
  ];

  initialPathEl.setAttribute('d', airfoilToPath(pts));

  if (optimizedFillEl) {
    optimizedFillEl.setAttribute('d', '');
    optimizedFillEl.style.opacity = 0;
  }

  if (optimizedStrokeEl) {
    optimizedStrokeEl.setAttribute('d', '');
    optimizedStrokeEl.style.opacity = 0;
  }

  if (flowMarkerEl) {
    flowMarkerEl.style.opacity = 0;
  }
}

function animateFlowMarker(pathEl, duration = 1400) {
  if (!pathEl || !flowMarkerEl) return;

  const totalLength = pathEl.getTotalLength();
  flowMarkerEl.style.opacity = 1;

  let start = null;

  function step(timestamp) {
    if (!start) start = timestamp;

    const progress = Math.min((timestamp - start) / duration, 1);
    const point = pathEl.getPointAtLength(totalLength * progress);

    flowMarkerEl.setAttribute('cx', point.x);
    flowMarkerEl.setAttribute('cy', point.y);

    if (progress < 1) {
      requestAnimationFrame(step);
    } else {
      setTimeout(() => {
        flowMarkerEl.style.opacity = 0;
      }, 250);
    }
  }

  requestAnimationFrame(step);
}

function animateOptimizedGeometry(geometry) {
  if (!geometry || !geometry.initial || !geometry.optimized) return;

  if (!initialPathEl || !optimizedFillEl || !optimizedStrokeEl) {
    console.error('SVG IDs are missing: initialAirfoil, optimizedAirfoilFill, optimizedAirfoilStroke, flowMarker');
    return;
  }

  const initPts = [
    ...geometry.initial.upper,
    ...[...geometry.initial.lower].reverse()
  ];

  const optPts = [
    ...geometry.optimized.upper,
    ...[...geometry.optimized.lower].reverse()
  ];

  initialPathEl.setAttribute('d', airfoilToPath(initPts));

  optimizedFillEl.setAttribute('d', airfoilToPath(optPts));
  optimizedStrokeEl.setAttribute('d', airfoilToPath(optPts));

  optimizedFillEl.style.opacity = 0;
  optimizedStrokeEl.style.opacity = 1;

  const totalLength = optimizedStrokeEl.getTotalLength();

  optimizedStrokeEl.style.transition = 'none';
  optimizedStrokeEl.style.strokeDasharray = totalLength;
  optimizedStrokeEl.style.strokeDashoffset = totalLength;
  optimizedStrokeEl.classList.remove('airfoil-reveal');

  optimizedStrokeEl.getBoundingClientRect();

  optimizedStrokeEl.style.transition = 'stroke-dashoffset 1200ms ease, opacity 350ms ease';
  optimizedStrokeEl.style.strokeDashoffset = '0';

  setTimeout(() => {
    optimizedFillEl.style.transition = 'opacity 700ms ease';
    optimizedFillEl.style.opacity = 1;
    optimizedStrokeEl.classList.add('airfoil-reveal');
  }, 250);

  animateFlowMarker(optimizedStrokeEl, 1400);
}

function renderMetrics(metrics) {
  const cards = [
    ['cl', 'C<sub>L</sub>', false],
    ['cd', 'C<sub>D</sub>', true],
    ['cl_cd', 'C<sub>L</sub>/C<sub>D</sub>', false],
    ['cm', 'C<sub>M</sub>', false],
    ['tc', 't/c', false]
  ];

  metricsEl.innerHTML = cards.map(([key, label, inverse]) => {
    return `
      <div class="metric-card">
        <div class="metric-label">${label}</div>
        <div class="metric-value">${formatMetric(key, metrics[key])}</div>
        <div class="metric-delta">${formatDelta(metrics[key], previousMetrics[key], inverse)}</div>
      </div>
    `;
  }).join('');
}

function renderConstraints(result) {
  const cm = result.constraints.cm;
  const tc = result.constraints.tc;

  constraintsEl.innerHTML = `
    <h3 class="panel-title with-icon">♡ Constraint Verification</h3>

    <div class="check-row ${cm.satisfied ? 'ok' : 'bad'}">
      <span class="round-icon ${cm.satisfied ? 'ok' : 'bad'}">${cm.satisfied ? '✓' : '✕'}</span>
      <div>
        Pitching Moment (C<sub>M</sub>)
        <span class="small-muted">
          C<sub>M</sub> = ${cm.value.toFixed(4)} ∈ [${cm.min}, ${cm.max}]
        </span>
      </div>
    </div>

    <div class="check-row ${tc.satisfied ? 'ok' : 'bad'}">
      <span class="round-icon ${tc.satisfied ? 'ok' : 'bad'}">${tc.satisfied ? '✓' : '✕'}</span>
      <div>
        Thickness Ratio (t/c)
        <span class="small-muted">
          t/c = ${(tc.value * 100).toFixed(1)}% ∈ [${(tc.min * 100).toFixed(0)}%, ${(tc.max * 100).toFixed(0)}%]
        </span>
      </div>
    </div>
  `;
}

function renderXAI(result) {
  const items = result.xai?.feature_importance || [];

  const safeItems = items.length
    ? items
    : [
        { feature: 'SHAP Values', score: 0 },
        { feature: 'LIME Impact', score: 0 },
        { feature: 'Feature Importance', score: 0 }
      ];

  xaiEl.innerHTML = `
    <h3 class="panel-title with-icon purple">◈ Model Explainability (XAI)</h3>
    ${safeItems.slice(0, 3).map((item, index) => {
      const label = index === 0 ? 'SHAP Values' : index === 1 ? 'LIME Impact' : item.feature;
      return `
        <div class="xai-row">
          <span>${label}</span>
          <strong>${Number(item.score).toFixed(3)}</strong>
        </div>
      `;
    }).join('')}
  `;
}

function renderDecisionLogic(result) {
  if (!decisionLogicEl) return;

  const logic = result.decision_logic || [
    'If C<sub>L</sub>/C<sub>D</sub> improves and constraints are satisfied → Accept geometry',
    'If C<sub>M</sub> leaves feasible range → Penalize pitching moment',
    'If t/c violates structural interval → Reject candidate'
  ];

  decisionLogicEl.innerHTML = `
    <h3 class="panel-title with-icon blue">☷ Decision Logic</h3>
    ${logic.map((item) => `
      <div class="logic-row">${item}</div>
    `).join('')}
  `;
}

function renderPipeline(result) {
  const steps = result.pipeline || [];

  pipelineEl.innerHTML = `
    <h3 class="panel-title with-icon">☑ Optimization Pipeline</h3>
    ${steps.slice(0, 5).map((step) => `
      <div class="pipeline-row done">✓ ${step}</div>
    `).join('')}
  `;
}

function renderExperiment(result) {
  const summary = result.experiment?.summary || 'Experiment summary will be generated after optimization.';

  experimentEl.innerHTML = `
    <h3 class="panel-title">Experiment Summary</h3>
    <p class="text-sm text-navy-600 mt-2 leading-relaxed">${summary}</p>
  `;
}

function renderResult(result) {
  if (!result || result.status !== 'ok') {
    throw new Error(result?.message || 'Optimization failed.');
  }

  renderMetrics(result.metrics);
  renderConstraints(result);
  renderXAI(result);
  renderDecisionLogic(result);
  renderPipeline(result);
  renderExperiment(result);
  animateOptimizedGeometry(result.geometry);

  previousMetrics = result.metrics;
}

function getPayload() {
  const selectedModel = document.querySelector('input[name="modelSelect"]:checked');

  if (!selectedModel) {
    throw new Error('Please select a DRL model.');
  }

  const upperWeights = parseWeights($('upper').value);
  const lowerWeights = parseWeights($('lower').value);

  if (upperWeights.length === 0) {
    throw new Error('Upper Surface Weights cannot be empty.');
  }

  if (lowerWeights.length === 0) {
    throw new Error('Lower Surface Weights cannot be empty.');
  }

  return {
    model: selectedModel.value,
    aoa: parseNumberInput('aoa', 'Angle of Attack'),
    reynolds: parseNumberInput('reynolds', 'Reynolds Number'),
    upper_weights: upperWeights,
    lower_weights: lowerWeights,
    leading_edge_weight: parseNumberInput('le', 'Leading Edge Weight'),
    trailing_edge_offset: parseNumberInput('te', 'Trailing Edge Offset')
  };
}

async function runOptimization() {
  const payload = getPayload();

  try {
    setButtonLoading(true);
    modelStatus.textContent = `⚙ ${payload.model} model analyzing...`;
    setBackendStatus('running', 'Running');

    const response = await fetch('/api/optimize/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.message || 'Backend error.');
    }

    renderResult(data);
    modelStatus.textContent = `⚙ ${payload.model} model ready`;
    setBackendStatus('ready', 'Model Ready');

  } catch (error) {
    console.error(error);
    modelStatus.textContent = error.message;
    setBackendStatus('error', 'Error');

  } finally {
    setButtonLoading(false);
  }
}

document.querySelectorAll('input[name="modelSelect"]').forEach((radio) => {
  radio.addEventListener('change', (event) => {
    modelStatus.textContent = `⚙ ${event.target.value} model loaded`;
  });
});

$('optimizeBtn').addEventListener('click', runOptimization);

window.addEventListener('DOMContentLoaded', () => {
  renderInitialOnly();
  setBackendStatus('ready', 'Model Ready');
});
