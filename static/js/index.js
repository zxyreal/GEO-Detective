function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button ? button.querySelector('.copy-text') : null;
  if (!bibtexElement || !button || !copyText) return;

  const markCopied = function () {
    button.classList.add('copied');
    copyText.textContent = 'Copied!';
    window.setTimeout(function () {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(bibtexElement.textContent).then(markCopied);
    return;
  }

  const textArea = document.createElement('textarea');
  textArea.value = bibtexElement.textContent;
  textArea.style.position = 'fixed';
  textArea.style.opacity = '0';
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand('copy');
  document.body.removeChild(textArea);
  markCopied();
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

const geoAccuracyData = {
  country: {
    label: 'Country',
    baseline: [74.0, 60.1, 35.5, 25.8],
    agent: [77.3, 57.7, 40.1, 28.9],
    insight: '<strong>Country:</strong> GEO-Detective gains +4.6 pp on Difficult and +3.1 pp on Very Difficult images, while Moderate drops by 2.4 pp.'
  },
  state: {
    label: 'State',
    baseline: [64.7, 44.1, 16.6, 15.5],
    agent: [68.8, 42.0, 18.3, 13.4],
    insight: '<strong>State:</strong> gains appear on Easy (+4.1 pp) and Difficult (+1.7 pp), but not on Moderate or Very Difficult images.'
  },
  city: {
    label: 'City',
    baseline: [58.0, 30.2, 9.7, 8.2],
    agent: [63.2, 28.8, 11.7, 9.3],
    insight: '<strong>City:</strong> the largest gain is +5.2 pp on Easy images; Difficult and Very Difficult improve by +2.0 and +1.1 pp.'
  }
};

const difficultyLabels = ['Easy', 'Moderate', 'Difficult', 'Very Difficult'];

function renderAccuracyChart(level) {
  const chart = document.getElementById('interactive-chart');
  const insight = document.getElementById('chart-insight');
  const data = geoAccuracyData[level];
  if (!chart || !insight || !data) return;

  chart.innerHTML = difficultyLabels.map(function (difficulty, index) {
    const baseline = data.baseline[index];
    const agent = data.agent[index];
    const delta = agent - baseline;
    const deltaLabel = (delta >= 0 ? '+' : '') + delta.toFixed(1);
    return '<div class="bar-group">' +
      '<div class="bar-pair">' +
        '<div class="bar-column"><span class="bar-value">' + baseline.toFixed(1) + '</span><div class="chart-bar baseline" style="height:' + baseline + '%" title="Baseline: ' + baseline.toFixed(1) + '%"></div></div>' +
        '<div class="bar-column"><span class="bar-value">' + agent.toFixed(1) + '</span><div class="chart-bar agent" style="height:' + agent + '%" title="GEO-Detective: ' + agent.toFixed(1) + '%; change ' + deltaLabel + ' percentage points"><span class="bar-delta">' + deltaLabel + '</span></div></div>' +
      '</div>' +
      '<div class="difficulty-label">' + difficulty + '</div>' +
    '</div>';
  }).join('');

  chart.setAttribute('aria-label', data.label + '-level accuracy comparison between the o3 baseline and GEO-Detective across four difficulty levels');
  insight.innerHTML = data.insight;
}

function initializeAccuracyChart() {
  const tabs = document.querySelectorAll('.chart-tab');
  if (!tabs.length) return;

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      tabs.forEach(function (item) {
        const active = item === tab;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
      });
      renderAccuracyChart(tab.dataset.level);
    });
  });

  renderAccuracyChart('country');
}

function initializeResultTables() {
  const tabs = document.querySelectorAll('.table-tab');
  const panels = document.querySelectorAll('.table-panel');
  if (!tabs.length || !panels.length) return;

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      const targetId = tab.dataset.table;
      tabs.forEach(function (item) {
        const active = item === tab;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
      });
      panels.forEach(function (panel) {
        const active = panel.id === targetId;
        panel.classList.toggle('is-active', active);
        panel.hidden = !active;
      });
    });
  });
}

document.addEventListener('DOMContentLoaded', function () {
  initializeAccuracyChart();
  initializeResultTables();
});

window.addEventListener('scroll', function () {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (!scrollButton) return;
  scrollButton.classList.toggle('visible', window.pageYOffset > 300);
});
