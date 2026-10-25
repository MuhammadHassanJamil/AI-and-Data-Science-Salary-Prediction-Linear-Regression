/**
 * AI & Data Science Salary Predictor - Frontend Logic
 * ====================================================
 * FULLY DYNAMIC: All model metadata (weights, categories, metrics)
 * are fetched from /api/model-info at startup. Nothing is hardcoded.
 * Predictions are made via /api/predict (server-side pipeline inference).
 */

// ── Global State (populated dynamically from server) ──
let MODEL = null;  // Will hold the full model metadata from /api/model-info

// ── DOM References ──
const jobTitleSelect = document.getElementById('job-title');
const experienceSelect = document.getElementById('experience-level');
const companyLocationSelect = document.getElementById('company-location');
const employeeResidenceSelect = document.getElementById('employee-residence');
const companySizeInput = document.getElementById('company-size');
const employmentTypeSelect = document.getElementById('employment-type');
const remoteRange = document.getElementById('remote-range');
const remoteLabel = document.getElementById('remote-label');
const workYearSelect = document.getElementById('work-year');
const predictBtn = document.getElementById('predict-btn');

const salaryValueEl = document.getElementById('salary-value');
const ciRangeEl = document.getElementById('ci-range');
const diffPctEl = document.getElementById('diff-pct');
const meterFillEl = document.getElementById('meter-fill');
const driverBreakdownEl = document.getElementById('driver-breakdown');
const modelBadgeEl = document.getElementById('model-badge');

// ── Initialize Application ──
window.addEventListener('DOMContentLoaded', async () => {
  await loadModelInfo();
  setupEventListeners();
  calculateSalary();
});

/**
 * Fetch model metadata from the server and populate all dropdowns dynamically.
 * This ensures NOTHING is hardcoded in the frontend.
 */
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    MODEL = await res.json();

    console.log('Model metadata loaded dynamically:', MODEL.algorithm);
    console.log(`  Features: ${MODEL.feature_names.length}`);
    console.log(`  CV RMSE: $${MODEL.cv_mean_rmse}`);
    console.log(`  Test R²: ${MODEL.test_r2}`);

    // Populate dropdowns from model categories
    populateSelect(jobTitleSelect, MODEL.categories.job_title, 'Machine Learning Engineer');
    populateSelect(experienceSelect, MODEL.categories.experience_level, 'Senior');
    populateSelect(companyLocationSelect, MODEL.categories.company_location, 'United States');
    populateSelect(employeeResidenceSelect, MODEL.categories.employee_residence, 'United States');
    populateSelect(employmentTypeSelect, MODEL.categories.employment_type, 'Full-time');

    // Populate company size buttons dynamically
    populateCompanySizeButtons(MODEL.categories.company_size);

    // Update model badge
    if (modelBadgeEl) {
      modelBadgeEl.textContent = MODEL.algorithm + ' Pipeline';
    }

    // Update stats banner with dynamic values
    updateStatsBanner();

    // Dynamically render the weights tab
    renderWeightsTab();

    // Update R² badge in header
    const r2Badge = document.getElementById('r2-badge');
    if (r2Badge) r2Badge.textContent = `Lasso Rank 1 (R² ${MODEL.test_r2.toFixed(2)})`;

    // Update average label in comparison bar
    const avgLabel = document.getElementById('avg-label');
    if (avgLabel) avgLabel.textContent = `$${Math.round(MODEL.industry_avg / 1000)}k (Avg)`;

  } catch (err) {
    console.error('Failed to load model info:', err);
    // Show error state
    salaryValueEl.textContent = 'Server Offline';
    ciRangeEl.textContent = 'Start server.py to enable predictions';
  }
}

/**
 * Populate a <select> element with options from the model categories.
 */
function populateSelect(selectEl, options, defaultValue) {
  selectEl.innerHTML = '';
  options.forEach(opt => {
    const optionEl = document.createElement('option');
    optionEl.value = opt;
    optionEl.textContent = opt;
    if (opt === defaultValue) optionEl.selected = true;
    selectEl.appendChild(optionEl);
  });
}

/**
 * Dynamically create company size segmented buttons from model categories.
 */
function populateCompanySizeButtons(sizes) {
  const container = document.getElementById('company-size-group');
  if (!container) return;

  container.innerHTML = '';
  const sizeLabels = {
    'Small': 'Small (<50)',
    'Medium': 'Medium (50-250)',
    'Large': 'Large (>250)'
  };

  sizes.forEach((size, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'segment-btn' + (size === 'Medium' ? ' active' : '');
    btn.dataset.value = size;
    btn.textContent = sizeLabels[size] || size;
    btn.addEventListener('click', () => {
      container.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      companySizeInput.value = size;
      calculateSalary();
    });
    container.appendChild(btn);
  });

  companySizeInput.value = 'Medium';
}

/**
 * Update the stats banner with real values from model metadata.
 */
function updateStatsBanner() {
  if (!MODEL) return;

  const el = (id) => document.getElementById(id);
  if (el('stat-instances')) el('stat-instances').textContent = (MODEL.train_size + MODEL.test_size).toLocaleString();
  if (el('stat-r2')) el('stat-r2').textContent = (MODEL.test_r2 * 100).toFixed(1) + '%';
  if (el('stat-rmse')) el('stat-rmse').textContent = '$' + Math.round(MODEL.cv_mean_rmse).toLocaleString();
}

/**
 * Dynamically render the Learned Model Weights tab from model metadata.
 */
function renderWeightsTab() {
  if (!MODEL || !MODEL.weights) return;

  const container = document.getElementById('weights-container');
  if (!container) return;

  const weights = MODEL.weights;
  const entries = Object.entries(weights).filter(([k, v]) => v !== 0);

  // Sort by value
  const positive = entries.filter(([k, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  const negative = entries.filter(([k, v]) => v < 0).sort((a, b) => a[1] - b[1]);

  // Helper to make readable labels from feature names
  function readableLabel(key) {
    return key
      .replace('cat__experience_level_', '')
      .replace('cat__employment_type_', '')
      .replace('cat__job_title_', '')
      .replace('cat__employee_residence_', 'Residence: ')
      .replace('cat__company_location_', 'Location: ')
      .replace('cat__company_size_', '')
      .replace('num__work_year', 'Work Year (std)')
      .replace('num__remote_ratio', 'Remote Ratio (std)');
  }

  function buildCard(title, items, cls) {
    let html = `<div class="weight-card"><h4>${title}</h4>`;
    items.slice(0, 8).forEach(([key, val]) => {
      const sign = val >= 0 ? '+' : '-';
      html += `<div class="weight-row"><span class="w-name">${readableLabel(key)}</span><span class="w-val ${cls}">${sign}$${Math.abs(Math.round(val)).toLocaleString()}</span></div>`;
    });
    html += '</div>';
    return html;
  }

  container.innerHTML =
    buildCard('Top Positive Drivers (Increases Salary)', positive, 'pos') +
    buildCard('Top Downward Drivers (Decreases Salary)', negative, 'neg');
}

// ── Event Listeners ──
function setupEventListeners() {
  // Remote range label update
  remoteRange.addEventListener('input', () => {
    const val = parseInt(remoteRange.value, 10);
    if (val === 100) remoteLabel.textContent = '100% Fully Remote';
    else if (val === 50) remoteLabel.textContent = '50% Hybrid';
    else remoteLabel.textContent = '0% On-site';
    calculateSalary();
  });

  // Select changes trigger automatic update
  [
    jobTitleSelect,
    experienceSelect,
    companyLocationSelect,
    employeeResidenceSelect,
    employmentTypeSelect,
    workYearSelect
  ].forEach(el => {
    el.addEventListener('change', calculateSalary);
  });

  predictBtn.addEventListener('click', calculateSalary);

  // Tabs functionality
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ── Prediction (Server-Side Only, No Hardcoded Fallback) ──
async function calculateSalary() {
  if (!MODEL) return;

  const inputData = {
    work_year: parseInt(workYearSelect.value, 10),
    remote_ratio: parseFloat(remoteRange.value),
    experience_level: experienceSelect.value,
    employment_type: employmentTypeSelect.value,
    job_title: jobTitleSelect.value,
    employee_residence: employeeResidenceSelect.value,
    company_location: companyLocationSelect.value,
    company_size: companySizeInput.value
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputData),
    });

    if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);

    const data = await res.json();
    let predictedSalary = data.predicted_salary;

    // Clip to sensible salary limits
    predictedSalary = Math.max(25000, Math.min(450000, predictedSalary));
    updateUI(predictedSalary, inputData, data);

  } catch (err) {
    console.error('Prediction error:', err);
    salaryValueEl.textContent = 'Error';
    ciRangeEl.textContent = 'Could not reach prediction server';
  }
}

// ── UI Rendering (uses dynamic MODEL metadata) ──
function updateUI(salary, input, apiResponse) {
  // Animated Counter
  animateValue(salaryValueEl, salary);

  // Confidence Interval — dynamically from API response
  const lowerCI = Math.max(25000, apiResponse.lower_ci_95);
  const upperCI = apiResponse.upper_ci_95;
  ciRangeEl.textContent = `Expected 95% CI: $${Math.round(lowerCI).toLocaleString()} — $${Math.round(upperCI).toLocaleString()}`;

  // Difference vs Industry Average (dynamically from model metadata)
  const industryAvg = MODEL.industry_avg;
  const diff = ((salary - industryAvg) / industryAvg) * 100;
  const diffSign = diff >= 0 ? '+' : '';
  diffPctEl.textContent = `${diffSign}${diff.toFixed(1)}%`;
  diffPctEl.className = `diff-tag ${diff >= 0 ? 'diff-pos' : 'diff-neg'}`;

  // Meter Bar (scaled between $25k and $300k)
  const pct = Math.min(100, Math.max(0, ((salary - 25000) / (300000 - 25000)) * 100));
  meterFillEl.style.width = `${pct}%`;

  // Render Driver Breakdown (from dynamic model weights)
  renderDrivers(input);
}

/**
 * Compute and display top salary drivers using dynamically loaded model weights.
 */
function renderDrivers(input) {
  if (!MODEL || !MODEL.weights) return;

  const weights = MODEL.weights;
  const drivers = [];

  // Experience level
  const expKey = `cat__experience_level_${input.experience_level}`;
  if (weights[expKey]) drivers.push({ name: `${input.experience_level} Experience`, val: weights[expKey] });

  // Company location
  const locKey = `cat__company_location_${input.company_location}`;
  if (weights[locKey]) drivers.push({ name: `${input.company_location} Market Rate`, val: weights[locKey] });

  // Job title
  const titleKey = `cat__job_title_${input.job_title}`;
  if (weights[titleKey]) drivers.push({ name: `${input.job_title} Premium`, val: weights[titleKey] });

  // Company size
  const sizeKey = `cat__company_size_${input.company_size}`;
  if (weights[sizeKey]) drivers.push({ name: `${input.company_size} Enterprise Scale`, val: weights[sizeKey] });

  // Remote effect (dynamically from model scales)
  const remoteScales = MODEL.numeric_scales.remote_ratio;
  const zRemote = (input.remote_ratio - remoteScales.mean) / remoteScales.scale;
  const remoteWeight = weights['num__remote_ratio'] || 0;
  const remoteVal = zRemote * remoteWeight;
  if (Math.abs(remoteVal) > 500) {
    drivers.push({ name: `${input.remote_ratio}% Remote Arrangement`, val: remoteVal });
  }

  // Sort by absolute impact
  drivers.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));

  driverBreakdownEl.innerHTML = '';
  drivers.slice(0, 4).forEach(item => {
    const isPos = item.val >= 0;
    const sign = isPos ? '+' : '-';
    const li = document.createElement('li');
    li.className = 'driver-item';
    li.innerHTML = `
      <span class="driver-name">${item.name}</span>
      <span class="driver-val ${isPos ? 'pos' : 'neg'}">${sign}$${Math.abs(Math.round(item.val)).toLocaleString()}</span>
    `;
    driverBreakdownEl.appendChild(li);
  });
}

// ── Animation Helper ──
function animateValue(obj, endValue) {
  const currentStr = obj.textContent.replace(/,/g, '');
  const start = parseInt(currentStr, 10) || endValue;
  const duration = 400;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (endValue - start) * easeProgress);

    obj.textContent = current.toLocaleString();

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}
