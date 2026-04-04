import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0';

const MODEL_ID = 'Xenova/bge-small-zh-v1.5';
const DATA_CSV_URL = './data/data.csv';
const EMBEDDINGS_CSV_URL = './data/embeddings.csv';
env.allowLocalModels = false;

const COLUMN_DEFS = [
  { key: 'id', label: 'ID', semantic: false, fixed: true, widthClass: 'col-id' },
  { key: '__score', label: '得分', semantic: false, fixed: true, widthClass: 'col-score' },
  { key: 'title', label: '标题', semantic: true, widthClass: 'col-title' },
  { key: 'template', label: '空白模板', semantic: true, widthClass: 'col-template' },
  { key: 'relationship', label: '角色关系', semantic: true, widthClass: 'col-relationship' },
  { key: 'cast_size', label: '人数', semantic: true, widthClass: 'col-cast-size' },
  { key: 'logline', label: '一句话定义', semantic: true, widthClass: 'col-logline' },
  { key: 'location', label: '地点', semantic: true, widthClass: 'col-location' },
  { key: 'forbidden_elements', label: '禁止使用元素', semantic: false, widthClass: 'col-forbidden' },
  { key: 'self_check', label: '自检问题', semantic: false, widthClass: 'col-self-check' },
  { key: 'examples', label: '示例', semantic: false, widthClass: 'col-example' }
];
const REQUIRED_DATA_COLUMNS = ['id', 'title', 'template', 'relationship', 'cast_size', 'logline', 'location', 'forbidden_elements', 'self_check', 'examples'];
const EMBEDDING_COLUMNS = ['title', 'template', 'relationship', 'cast_size', 'logline', 'location'];
const SEARCHABLE_COLUMNS = COLUMN_DEFS.filter(col => !['id', '__score'].includes(col.key));
const SEMANTIC_COLUMNS = new Set(COLUMN_DEFS.filter(col => col.semantic).map(col => col.key));

const STORAGE_KEY_VISIBLE = 'notion_csv_visible_columns_v1';
const STORAGE_KEY_THRESHOLD = 'notion_csv_default_threshold_v4';
const STORAGE_KEY_GLOBAL_SEARCH = 'notion_csv_global_search_v1';
const STORAGE_KEY_SORT = 'notion_csv_sort_column_v1';
const DEFAULT_SEMANTIC_THRESHOLD = 0.5;

let rawData = [];
let embeddingMap = new Map();
let extractor = null;
let visibleColumns = loadVisibleColumns();
let defaultSemanticThreshold = loadDefaultThreshold();
let globalSearchValue = loadGlobalSearch();
let sortColumnKey = loadSortColumn();
let globalSearchTimer = null;
let openColumnMenu = null;
let openPropertiesMenu = false;
let openSortMenu = false;
let openSearchMenu = false;
let queryVectors = {};
let filteredRows = [];
let isApplying = false;

const TAG_STYLE_COLUMNS = new Set(['relationship', 'cast_size', 'location', 'forbidden_elements', 'self_check']);
const columnDraftState = Object.fromEntries(SEARCHABLE_COLUMNS.map(col => [col.key, '']));

const columnState = Object.fromEntries(SEARCHABLE_COLUMNS.map(col => [col.key, {
  query: '',
  threshold: defaultSemanticThreshold,
}]));

const els = {
  thead: document.getElementById('thead'),
  tbody: document.getElementById('tbody'),
  status: document.getElementById('status'),
  countPill: document.getElementById('countPill'),
  activePill: document.getElementById('activePill'),
  sortPill: document.getElementById('sortPill'),
  devicePill: document.getElementById('devicePill'),
  initBtn: document.getElementById('initBtn'),
  reloadBtn: document.getElementById('reloadBtn'),
  clearBtn: document.getElementById('clearBtn'),
  propertiesBtn: document.getElementById('propertiesBtn'),
  sortBtn: document.getElementById('sortBtn'),
  searchBtn: document.getElementById('searchBtn'),
  menuLayer: document.getElementById('menuLayer')
};
const MENU_BUTTON_SELECTORS = ['#propertiesBtn', '#sortBtn', '#searchBtn'];

function loadVisibleColumns() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY_VISIBLE) || 'null');
    if (Array.isArray(raw) && raw.length) return raw;
  } catch {}
  return SEARCHABLE_COLUMNS.map(col => col.key);
}
function loadDefaultThreshold() {
  const value = Number(localStorage.getItem(STORAGE_KEY_THRESHOLD) || String(DEFAULT_SEMANTIC_THRESHOLD));
  return Number.isFinite(value) ? Math.min(0.95, Math.max(0, value)) : DEFAULT_SEMANTIC_THRESHOLD;
}
function loadGlobalSearch() {
  return localStorage.getItem(STORAGE_KEY_GLOBAL_SEARCH) || '';
}
function loadSortColumn() {
  const value = localStorage.getItem(STORAGE_KEY_SORT) || '';
  return EMBEDDING_COLUMNS.includes(value) ? value : '';
}
function saveVisibleColumns() { localStorage.setItem(STORAGE_KEY_VISIBLE, JSON.stringify(visibleColumns)); }
function saveDefaultThreshold() { localStorage.setItem(STORAGE_KEY_THRESHOLD, String(defaultSemanticThreshold)); }
function saveGlobalSearch() { localStorage.setItem(STORAGE_KEY_GLOBAL_SEARCH, globalSearchValue); }
function saveSortColumn() {
  if (sortColumnKey) localStorage.setItem(STORAGE_KEY_SORT, sortColumnKey);
  else localStorage.removeItem(STORAGE_KEY_SORT);
}
function setStatus(text) { els.status.textContent = text; }
function closeAllMenus() {
  openColumnMenu = null;
  openPropertiesMenu = false;
  openSortMenu = false;
  openSearchMenu = false;
}
function hasOpenMenu() {
  return !!(openColumnMenu || openPropertiesMenu || openSortMenu || openSearchMenu);
}
function toggleToolbarMenu(menuKey) {
  const wasOpen = menuKey === 'properties' ? openPropertiesMenu
    : menuKey === 'sort' ? openSortMenu
    : openSearchMenu;
  closeAllMenus();
  if (menuKey === 'properties') openPropertiesMenu = !wasOpen;
  if (menuKey === 'sort') openSortMenu = !wasOpen;
  if (menuKey === 'search') openSearchMenu = !wasOpen;
}
function renderUI() {
  renderHeader();
  renderBody();
  renderMenus();
  updateToolbarState();
}

function escapeHtml(str = '') {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function normalizeText(value) { return String(value ?? '').replace(/\s+/g, ' ').trim(); }
function cleanCellText(value) { return String(value ?? '').trim(); }
function isVisible(key) { return key === 'id' || key === '__score' || visibleColumns.includes(key); }
function getColumnLabel(key) { return COLUMN_DEFS.find(col => col.key === key)?.label || key; }
function hasActiveSemanticQuery(key) { return SEMANTIC_COLUMNS.has(key) && !!normalizeText(columnState[key]?.query); }
function dot(a, b) { let s = 0; const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
function cosineSimilarity(a, b) { if (!a || !b || !a.length || !b.length) return 0; return dot(a, b); }

function splitVideoLinks(value) {
  const source = String(value || '');
  const matches = source.match(/https?:\/\/[^\s|；;，,]+/g);
  if (matches && matches.length) return matches.map(v => v.trim()).filter(Boolean);
  return source.split(/[\n|｜;；，,]+/).map(v => v.trim()).filter(Boolean);
}
function splitTagValues(value) {
  return String(value || '').split(/[|｜;；]+/).map(v => v.trim()).filter(Boolean);
}
function getYouTubeEmbedUrl(url) {
  try {
    const u = new URL(url);
    if (u.hostname.includes('youtu.be')) {
      const id = u.pathname.replace(/^\//, '');
      return id ? `https://www.youtube.com/embed/${id}` : null;
    }
    if (u.hostname.includes('youtube.com')) {
      const id = u.searchParams.get('v');
      if (id) return `https://www.youtube.com/embed/${id}`;
      const parts = u.pathname.split('/').filter(Boolean);
      const shortsIndex = parts.indexOf('shorts');
      if (shortsIndex >= 0 && parts[shortsIndex + 1]) return `https://www.youtube.com/embed/${parts[shortsIndex + 1]}`;
      const embedIndex = parts.indexOf('embed');
      if (embedIndex >= 0 && parts[embedIndex + 1]) return `https://www.youtube.com/embed/${parts[embedIndex + 1]}`;
    }
  } catch {}
  return null;
}

function parseCSV(text) {
  const rows = [];
  let row = [], cell = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i], next = text[i + 1];
    if (char === '"') {
      if (inQuotes && next === '"') { cell += '"'; i++; } else { inQuotes = !inQuotes; }
    } else if (char === ',' && !inQuotes) {
      row.push(cell); cell = '';
    } else if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i++;
      row.push(cell);
      if (row.some(v => v !== '')) rows.push(row);
      row = []; cell = '';
    } else {
      cell += char;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    if (row.some(v => v !== '')) rows.push(row);
  }
  return rows;
}
function rowsToObjects(rows) {
  if (!rows.length) return [];
  const headers = rows[0].map(h => normalizeText(h));
  return rows.slice(1).map(r => {
    const obj = {};
    headers.forEach((header, index) => { obj[header] = String(r[index] ?? ''); });
    return obj;
  });
}
function validateDataColumns(objects) {
  if (!objects.length) throw new Error('data.csv 里没有数据行。');
  const missing = REQUIRED_DATA_COLUMNS.filter(col => !(col in objects[0]));
  if (missing.length) throw new Error(`data.csv 缺少列：${missing.join('、')}`);
}
function parseEmbeddingCell(value) {
  const text = String(value ?? '').trim();
  if (!text) return null;
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed) && parsed.length) return parsed.map(v => Number(v)).filter(v => Number.isFinite(v));
  } catch {}
  return null;
}
function parseDataCsv(text) {
  const objects = rowsToObjects(parseCSV(text)).map(item => {
    const row = {};
    for (const col of REQUIRED_DATA_COLUMNS) row[col] = cleanCellText(item[col] ?? '');
    return row;
  });
  validateDataColumns(objects);
  return objects;
}
function parseEmbeddingsCsv(text) {
  const objects = rowsToObjects(parseCSV(text));
  if (!objects.length) return new Map();
  const headers = Object.keys(objects[0]);
  const missing = EMBEDDING_COLUMNS.filter(col => !(headers.includes(col) || headers.includes(`${col}__embedding`)));
  if (!headers.includes('id')) missing.unshift('id');
  if (missing.length) throw new Error(`embeddings.csv 缺少列：${missing.join('、')}`);
  const map = new Map();
  for (const item of objects) {
    const id = normalizeText(item.id);
    if (!id) continue;
    const record = { id };
    for (const col of EMBEDDING_COLUMNS) {
      const raw = item[col] ?? item[`${col}__embedding`];
      const vector = parseEmbeddingCell(raw);
      if (vector?.length >= 16) record[col] = vector;
    }
    map.set(id, record);
  }
  return map;
}
async function fetchCsv(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url} 读取失败（${response.status}）`);
  return response.text();
}
async function loadCsvData() {
  setStatus('正在读取 data.csv 和 embeddings.csv ...');
  const [dataResult, embeddingResult] = await Promise.allSettled([
    fetchCsv(DATA_CSV_URL),
    fetchCsv(EMBEDDINGS_CSV_URL),
  ]);
  if (dataResult.status !== 'fulfilled') {
    throw new Error('data.csv 无法读取。若你是直接双击 HTML 用 file:// 打开，浏览器通常会拦截 fetch；请放到本地静态服务器或站点里访问。');
  }
  rawData = parseDataCsv(dataResult.value).map((row, index) => ({ ...row, __rowIndex: index }));
  embeddingMap = embeddingResult.status === 'fulfilled' ? parseEmbeddingsCsv(embeddingResult.value) : new Map();
  for (const def of SEARCHABLE_COLUMNS) {
    columnDraftState[def.key] = columnState[def.key].query || '';
  }
  if (sortColumnKey && !hasActiveSemanticQuery(sortColumnKey)) sortColumnKey = '';
  filteredRows = rawData.map(row => ({ ...row, __semanticScores: {} }));
  updateSummary();
  renderUI();

  const linked = Array.from(embeddingMap.values()).filter(item => EMBEDDING_COLUMNS.some(col => Array.isArray(item[col]) && item[col].length)).length;
  if (embeddingResult.status === 'fulfilled') {
    setStatus(`已读取 ${rawData.length} 条数据，embeddings.csv 关联到 ${linked} 条 id。`);
  } else {
    setStatus(`已读取 ${rawData.length} 条数据，但 embeddings.csv 未读到；语义搜索暂时不可用。`);
  }
}

async function initModel() {
  if (extractor) return extractor;
  setStatus('正在初始化模型，首次会下载权重。');
  els.devicePill.textContent = 'WASM';
  extractor = await pipeline('feature-extraction', MODEL_ID, { device: 'wasm', dtype: 'q8' });
  setStatus('模型初始化完成。');
  return extractor;
}
async function embedQuery(text) {
  const normalized = normalizeText(text);
  if (!normalized) return null;
  await initModel();
  const output = await extractor(normalized, { pooling: 'mean', normalize: true });
  return typeof output.tolist === 'function' ? output.tolist()[0] : Array.from(output.data || []);
}

function getRenderedColumns() { return COLUMN_DEFS.filter(col => isVisible(col.key)); }
function renderHeader() {
  const columns = getRenderedColumns();
  els.thead.innerHTML = `<tr>${columns.map(col => renderHeaderCell(col)).join('')}</tr>`;
  els.thead.querySelectorAll('[data-col-head]').forEach(btn => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      const key = btn.dataset.colHead;
      closeAllMenus();
      openColumnMenu = openColumnMenu === key ? null : key;
      renderMenus();
      updateToolbarState();
    });
  });
}
function renderHeaderCell(col) {
  if (col.key === 'id') return `<th class="${col.widthClass || ''}"><button class="header-btn"><div class="header-main"><div class="header-name">#</div><div class="header-sub">固定列</div></div></button></th>`;
  if (col.key === '__score') {
    const label = sortColumnKey && hasActiveSemanticQuery(sortColumnKey) ? `${getColumnLabel(sortColumnKey)} 相关度` : '当前未排序';
    return `<th class="${col.widthClass || ''}"><button class="header-btn"><div class="header-main"><div class="header-name">得分</div><div class="header-sub">${label}</div></div></button></th>`;
  }
  const state = columnState[col.key];
  const hasQuery = !!normalizeText(state?.query);
  const label = col.semantic ? `语义 ${state.threshold.toFixed(2)}` : '关键词';
  return `
    <th class="${col.widthClass || ''}">
      <button class="header-btn ${openColumnMenu === col.key ? 'active' : ''}" data-col-head="${escapeHtml(col.key)}">
        <div class="header-main">
          <div class="header-name">
            <span class="header-icon">≡</span>
            <span>${escapeHtml(col.label)}</span>
            ${hasQuery ? '<span class="header-badge active">已筛选</span>' : ''}
          </div>
          <div class="header-sub"><span>${label}</span></div>
        </div>
        <span class="header-arrow">▾</span>
      </button>
    </th>`;
}
function renderBody() {
  const columns = getRenderedColumns();
  if (!filteredRows.length) {
    els.tbody.innerHTML = `<tr class="empty-row"><td colspan="${columns.length}">没有命中结果</td></tr>`;
    return;
  }
  els.tbody.innerHTML = filteredRows.map(row => `<tr>${columns.map(col => renderCellByColumn(row, col)).join('')}</tr>`).join('');
}
function renderCellByColumn(row, col) {
  const cls = col.widthClass || '';
  if (col.key === 'id') return `<td class="${cls}">${escapeHtml(row.id)}</td>`;
  if (col.key === '__score') {
    const hasScore = sortColumnKey && hasActiveSemanticQuery(sortColumnKey) && Number.isFinite(row.__semanticScores?.[sortColumnKey]);
    const score = hasScore ? row.__semanticScores?.[sortColumnKey] : 0;
    const sub = row.__semanticScores || {};
    return `<td class="${cls} score-cell"><span class="score-box">${hasScore ? score.toFixed(3) : '—'}</span>${Object.keys(sub).length ? `<div class="sub-scores">${Object.entries(sub).map(([k, v]) => `<span class="sub-score">${escapeHtml(getColumnLabel(k))} ${v.toFixed(3)}</span>`).join('')}</div>` : ''}</td>`;
  }
  return `<td class="${cls}">${renderCellValue(col.key, row[col.key])}</td>`;
}
function renderCellValue(columnKey, value) {
  if (columnKey === 'examples') {
    const links = splitVideoLinks(value);
    if (!links.length) return '<div class="cell-text"></div>';
    return `<div class="videos">${links.map(link => {
      const embed = getYouTubeEmbedUrl(link);
      return `<div class="video-item">${embed ? `<div class="video-frame"><iframe src="${escapeHtml(embed)}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>` : ''}<a class="video-link" href="${escapeHtml(link)}" target="_blank" rel="noreferrer">${escapeHtml(link)}</a></div>`;
    }).join('')}</div>`;
  }
  const tags = TAG_STYLE_COLUMNS.has(columnKey) ? splitTagValues(value) : [];
  if (tags.length) {
    const listClass = columnKey === 'forbidden_elements' ? 'tag-list tag-list-block' : 'tag-list';
    const chipClass = columnKey === 'forbidden_elements' ? 'tag-chip tag-chip-block' : `tag-chip ${TAG_STYLE_COLUMNS.has(columnKey) ? '' : 'soft'}`.trim();
    return `<div class="${listClass}">${tags.map(tag => `<span class="${chipClass}">${escapeHtml(tag)}</span>`).join('')}</div>`;
  }
  return `<div class="cell-text">${escapeHtml(value || '')}</div>`;
}

function getActiveFilters() { return SEARCHABLE_COLUMNS.filter(col => normalizeText(columnState[col.key].query)); }
function updateSummary() {
  const active = getActiveFilters().length + (normalizeText(globalSearchValue) ? 1 : 0);
  const activeSort = sortColumnKey && hasActiveSemanticQuery(sortColumnKey) ? sortColumnKey : '';
  els.countPill.textContent = `${filteredRows.length}/${rawData.length} 条`;
  els.activePill.textContent = `${active} 个筛选`;
  els.sortPill.textContent = activeSort ? `排序：${getColumnLabel(activeSort)}` : '未指定排序列';
}
function updateToolbarState() {
  els.propertiesBtn.classList.toggle('active', openPropertiesMenu);
  els.sortBtn.classList.toggle('active', openSortMenu);
  els.searchBtn.classList.toggle('active', openSearchMenu);
}
function anchorPosition(rect, width = 320) {
  const margin = 8; let left = rect.left, top = rect.bottom + 6;
  if (left + width > window.innerWidth - margin) left = window.innerWidth - width - margin;
  if (left < margin) left = margin;
  if (top > window.innerHeight - 140) top = Math.max(margin, rect.top - 12);
  return { left, top };
}
function renderMenus() {
  const parts = [];
  if (openColumnMenu) parts.push(renderColumnMenu(openColumnMenu));
  if (openPropertiesMenu) parts.push(renderPropertiesMenu());
  if (openSortMenu) parts.push(renderSortMenu());
  if (openSearchMenu) parts.push(renderSearchMenu());
  els.menuLayer.innerHTML = parts.join('');
  bindMenuEvents();
}
function renderColumnMenu(columnKey) {
  const btn = document.querySelector(`[data-col-head="${CSS.escape(columnKey)}"]`);
  if (!btn) return '';
  const rect = btn.getBoundingClientRect(), pos = anchorPosition(rect, 340);
  const def = COLUMN_DEFS.find(col => col.key === columnKey), state = columnState[columnKey];
  const draft = columnDraftState[columnKey] ?? state.query;
  return `
    <div class="menu" style="left:${pos.left}px; top:${pos.top}px; width:340px">
      <div class="menu-title">${escapeHtml(def.label)}</div>
      <div class="menu-section">
        <label class="menu-label">${def.semantic ? '语义搜索' : '关键词搜索'}</label>
        ${def.semantic ? `<textarea class="menu-textarea" data-role="column-query" data-column="${escapeHtml(columnKey)}" placeholder="只搜索这一列">${escapeHtml(draft)}</textarea>` : `<input class="menu-input" data-role="column-query" data-column="${escapeHtml(columnKey)}" type="search" placeholder="只搜索这一列" value="${escapeHtml(draft)}" />`}
      </div>
      ${def.semantic ? `<div class="menu-section"><label class="menu-label">相似度阈值</label><div class="menu-range-wrap"><input class="menu-range" data-role="column-threshold" data-column="${escapeHtml(columnKey)}" type="range" min="0" max="0.95" step="0.01" value="${state.threshold.toFixed(2)}" /><span class="menu-range-value" data-role="column-threshold-value" data-column="${escapeHtml(columnKey)}">${state.threshold.toFixed(2)}</span></div></div>` : ''}
      <div class="menu-section">
        <div class="menu-actions">
          <button class="menu-item primary" data-role="apply-column-filter" data-column="${escapeHtml(columnKey)}">应用搜索</button>
          <button class="menu-item" data-role="clear-column-filter" data-column="${escapeHtml(columnKey)}">清空本列</button>
        </div>
      </div>
      <div class="menu-section">
        <button class="menu-item" data-role="toggle-column-visibility" data-column="${escapeHtml(columnKey)}"><span class="left">隐藏这一列</span><span class="right">👁</span></button>
      </div>
      <div class="menu-section"><div class="legend">${def.semantic ? '先点“应用搜索”生成该列查询向量，再参与语义筛选和排序。' : '该列使用普通包含匹配；点“应用搜索”后生效。'}</div></div>
    </div>`;
}
function renderPropertiesMenu() {
  const rect = els.propertiesBtn.getBoundingClientRect(), pos = anchorPosition(rect, 320);
  return `
    <div class="menu" style="left:${pos.left}px; top:${pos.top}px">
      <div class="menu-title">属性可见性</div>
      <div class="menu-section">
        ${SEARCHABLE_COLUMNS.map(col => `<label class="menu-check"><span class="left"><input type="checkbox" data-role="visible-column" data-column="${escapeHtml(col.key)}" ${isVisible(col.key) ? 'checked' : ''} /><span>${escapeHtml(col.label)}</span></span><span class="right">${col.semantic ? '语义' : '文本'}</span></label>`).join('')}
      </div>
      <div class="menu-section">
        <label class="menu-label">默认语义阈值</label>
        <div class="menu-range-wrap">
          <input class="menu-range" id="defaultThresholdRange" type="range" min="0" max="0.95" step="0.01" value="${defaultSemanticThreshold.toFixed(2)}" />
          <span class="menu-range-value" id="defaultThresholdValue">${defaultSemanticThreshold.toFixed(2)}</span>
        </div>
      </div>
    </div>`;
}
function renderSortMenu() {
  const rect = els.sortBtn.getBoundingClientRect(), pos = anchorPosition(rect, 320);
  return `
    <div class="menu" style="left:${pos.left}px; top:${pos.top}px">
      <div class="menu-title">关联性排序</div>
      <div class="menu-section">
        <label class="menu-check"><span class="left"><input type="radio" name="sort-column" data-role="sort-column" value="" ${!sortColumnKey ? 'checked' : ''} /><span>不排序</span></span><span class="right">原顺序</span></label>
        ${EMBEDDING_COLUMNS.map(key => {
          const active = hasActiveSemanticQuery(key) && !!queryVectors[key];
          const checked = sortColumnKey === key ? 'checked' : '';
          const disabled = active ? '' : 'disabled';
          const hint = active ? '可用' : '先填搜索词';
          return `<label class="menu-check ${active ? '' : 'disabled'}"><span class="left"><input type="radio" name="sort-column" data-role="sort-column" value="${escapeHtml(key)}" ${checked} ${disabled} /><span>${escapeHtml(getColumnLabel(key))}</span></span><span class="right">${hint}</span></label>`;
        }).join('')}
      </div>
    </div>`;
}
function renderSearchMenu() {
  const rect = els.searchBtn.getBoundingClientRect(), pos = anchorPosition(rect, 360);
  return `
    <div class="menu" style="left:${pos.left}px; top:${pos.top}px; width:360px">
      <div class="menu-title">全局快速搜索</div>
      <div class="menu-section">
        <label class="menu-label">对当前可见列做普通文本搜索</label>
        <input class="menu-input" id="globalSearchInput" type="search" placeholder="例如：便利店 / 2人 / 暴雨" value="${escapeHtml(globalSearchValue)}" />
      </div>
      <div class="menu-section"><div class="legend">这是额外的文本筛选，不会替代单列语义搜索。</div></div>
    </div>`;
}
async function applyColumnFilter(columnKey) {
  const draft = normalizeText(columnDraftState[columnKey] ?? '');
  const previousQuery = columnState[columnKey].query;
  const previousVector = queryVectors[columnKey];
  try {
    if (SEMANTIC_COLUMNS.has(columnKey)) {
      if (!draft) {
        columnState[columnKey].query = '';
        delete queryVectors[columnKey];
      } else {
        setStatus(`正在为 ${getColumnLabel(columnKey)} 生成查询向量 ...`);
        const vector = await embedQuery(draft);
        columnState[columnKey].query = draft;
        queryVectors[columnKey] = vector;
      }
    } else {
      columnState[columnKey].query = draft;
    }
    if (!normalizeText(columnState[columnKey].query) && sortColumnKey === columnKey) {
      sortColumnKey = '';
      saveSortColumn();
    }
    await applyFilters();
    renderHeader();
    renderMenus();
  } catch (error) {
    columnState[columnKey].query = previousQuery;
    if (previousVector) queryVectors[columnKey] = previousVector;
    else delete queryVectors[columnKey];
    setStatus(`${getColumnLabel(columnKey)} 语义搜索失败：${error.message}`);
  }
}
function bindMenuEvents() {
  els.menuLayer.querySelectorAll('[data-role="column-query"]').forEach(input => {
    input.addEventListener('input', (event) => {
      const col = event.target.dataset.column;
      columnDraftState[col] = event.target.value;
    });
    input.addEventListener('keydown', async (event) => {
      const col = event.target.dataset.column;
      if (event.key === 'Enter' && (!SEMANTIC_COLUMNS.has(col) || event.metaKey || event.ctrlKey || !event.target.matches('textarea'))) {
        event.preventDefault();
        await applyColumnFilter(col);
      }
    });
  });
  els.menuLayer.querySelectorAll('[data-role="apply-column-filter"]').forEach(btn => {
    btn.addEventListener('click', async (event) => {
      event.stopPropagation();
      await applyColumnFilter(btn.dataset.column);
    });
  });
  els.menuLayer.querySelectorAll('[data-role="column-threshold"]').forEach(input => {
    input.addEventListener('input', (event) => {
      const col = event.target.dataset.column, value = Number(event.target.value);
      columnState[col].threshold = value;
      const out = els.menuLayer.querySelector(`[data-role="column-threshold-value"][data-column="${CSS.escape(col)}"]`);
      if (out) out.textContent = value.toFixed(2);
    });
    input.addEventListener('change', async () => {
      await applyFilters();
      renderHeader();
    });
  });
  els.menuLayer.querySelectorAll('[data-role="clear-column-filter"]').forEach(btn => {
    btn.addEventListener('click', async (event) => {
      event.stopPropagation();
      const col = btn.dataset.column;
      columnDraftState[col] = '';
      columnState[col].query = '';
      columnState[col].threshold = defaultSemanticThreshold;
      delete queryVectors[col];
      if (sortColumnKey === col) { sortColumnKey = ''; saveSortColumn(); }
      await applyFilters(); renderHeader(); renderMenus();
    });
  });
  els.menuLayer.querySelectorAll('[data-role="toggle-column-visibility"]').forEach(btn => {
    btn.addEventListener('click', (event) => { event.stopPropagation(); toggleVisibleColumn(btn.dataset.column, false); });
  });
  els.menuLayer.querySelectorAll('[data-role="visible-column"]').forEach(box => {
    box.addEventListener('change', (event) => toggleVisibleColumn(event.target.dataset.column, event.target.checked));
  });
  els.menuLayer.querySelectorAll('[data-role="sort-column"]').forEach(box => {
    box.addEventListener('change', async (event) => {
      sortColumnKey = event.target.value || '';
      saveSortColumn();
      await applyFilters();
      renderBody();
      updateSummary();
    });
  });
  const thresholdRange = document.getElementById('defaultThresholdRange');
  if (thresholdRange) {
    thresholdRange.addEventListener('input', (event) => {
      defaultSemanticThreshold = Number(event.target.value);
      const out = document.getElementById('defaultThresholdValue');
      if (out) out.textContent = defaultSemanticThreshold.toFixed(2);
      saveDefaultThreshold();
    });
  }
  const globalSearchInput = document.getElementById('globalSearchInput');
  if (globalSearchInput) {
    globalSearchInput.addEventListener('input', () => {
      globalSearchValue = globalSearchInput.value;
      saveGlobalSearch();
      clearTimeout(globalSearchTimer);
      globalSearchTimer = setTimeout(() => applyFilters(), 180);
    });
  }
}
function toggleVisibleColumn(columnKey, forceVisible) {
  const currentlyVisible = isVisible(columnKey);
  const nextVisible = typeof forceVisible === 'boolean' ? forceVisible : !currentlyVisible;
  if (nextVisible && !currentlyVisible) visibleColumns.push(columnKey);
  else if (!nextVisible && currentlyVisible) {
    if (visibleColumns.length <= 1) return;
    visibleColumns = visibleColumns.filter(key => key !== columnKey);
    if (openColumnMenu === columnKey) openColumnMenu = null;
    if (sortColumnKey === columnKey) { sortColumnKey = ''; saveSortColumn(); }
  }
  visibleColumns = SEARCHABLE_COLUMNS.map(col => col.key).filter(key => visibleColumns.includes(key));
  saveVisibleColumns();
  updateSummary();
  renderUI();
}

async function applyFilters() {
  if (isApplying) return;
  isApplying = true;
  try {
    const activeFilters = getActiveFilters();
    const activeGlobal = normalizeText(globalSearchValue);
    let hasMissingEmbeddings = false;
    const activeSort = sortColumnKey && hasActiveSemanticQuery(sortColumnKey) ? sortColumnKey : '';

    filteredRows = rawData.map(row => {
      const semanticScores = {};
      let passed = true;

      for (const def of activeFilters) {
        const query = normalizeText(columnState[def.key].query);
        if (!query) continue;
        const value = normalizeText(row[def.key]);
        let matched = false;

        if (def.semantic) {
          const emb = embeddingMap.get(String(row.id))?.[def.key];
          const queryVector = queryVectors[def.key];
          if (!emb || !queryVector) {
            hasMissingEmbeddings = true;
          } else {
            const score = cosineSimilarity(queryVector, emb);
            semanticScores[def.key] = score;
            matched = score >= columnState[def.key].threshold;
          }
        } else {
          matched = value.toLowerCase().includes(query.toLowerCase());
        }

        if (!matched) {
          passed = false;
          break;
        }
      }

      if (!passed) return null;

      if (activeGlobal) {
        const haystack = SEARCHABLE_COLUMNS.filter(col => isVisible(col.key)).map(col => normalizeText(row[col.key])).join(' ').toLowerCase();
        if (!haystack.includes(activeGlobal.toLowerCase())) return null;
      }

      return { ...row, __semanticScores: semanticScores };
    }).filter(Boolean);

    if (activeSort) {
      filteredRows.sort((a, b) => {
        const diff = (b.__semanticScores?.[activeSort] || 0) - (a.__semanticScores?.[activeSort] || 0);
        return diff || ((a.__rowIndex ?? 0) - (b.__rowIndex ?? 0));
      });
    } else {
      filteredRows.sort((a, b) => (a.__rowIndex ?? 0) - (b.__rowIndex ?? 0));
    }

    updateSummary();
    renderBody();

    if (!activeFilters.length && !activeGlobal) {
      setStatus(`未启用任何筛选。当前已加载 ${rawData.length} 条数据。`);
    } else if (hasMissingEmbeddings) {
      setStatus('部分语义列缺少 embedding 或对应 id 未命中，因此这些行无法参与该列语义筛选。');
    } else {
      setStatus(`已完成筛选，命中 ${filteredRows.length} 条。`);
    }
  } finally {
    isApplying = false;
  }
}

function clearAllFilters() {
  for (const def of SEARCHABLE_COLUMNS) {
    columnState[def.key].query = '';
    columnState[def.key].threshold = defaultSemanticThreshold;
    columnDraftState[def.key] = '';
  }
  queryVectors = {};
  globalSearchValue = '';
  sortColumnKey = '';
  saveGlobalSearch();
  saveSortColumn();
  closeAllMenus();
  filteredRows = rawData.map(row => ({ ...row, __semanticScores: {} }));
  updateSummary();
  renderUI();
  setStatus('已清空所有筛选。');
}
function closeMenus() {
  if (!hasOpenMenu()) return;
  closeAllMenus();
  renderMenus();
  updateToolbarState();
}

els.initBtn.addEventListener('click', async () => {
  try { await initModel(); }
  catch (error) { els.devicePill.textContent = '初始化失败'; setStatus(`模型初始化失败：${error.message}`); }
});
els.reloadBtn.addEventListener('click', async () => {
  try { await loadCsvData(); await applyFilters(); }
  catch (error) { setStatus(error.message); }
});
els.clearBtn.addEventListener('click', clearAllFilters);
els.propertiesBtn.addEventListener('click', (event) => { event.stopPropagation(); toggleToolbarMenu('properties'); renderMenus(); updateToolbarState(); });
els.sortBtn.addEventListener('click', (event) => { event.stopPropagation(); toggleToolbarMenu('sort'); renderMenus(); updateToolbarState(); });
els.searchBtn.addEventListener('click', (event) => { event.stopPropagation(); toggleToolbarMenu('search'); renderMenus(); updateToolbarState(); });

document.addEventListener('click', (event) => {
  if (event.target.closest('.menu')) return;
  if (event.target.closest('[data-col-head]')) return;
  if (MENU_BUTTON_SELECTORS.some(selector => event.target.closest(selector))) return;
  closeMenus();
});
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeMenus(); });
window.addEventListener('resize', closeMenus);
document.querySelector('.table-wrap').addEventListener('scroll', closeMenus);

async function bootstrap() {
  filteredRows = [];
  updateSummary();
  renderUI();
  try {
    await loadCsvData();
    await applyFilters();
  } catch (error) {
    setStatus(error.message);
    els.tbody.innerHTML = `<tr class="empty-row"><td colspan="${getRenderedColumns().length || 1}">${escapeHtml(error.message)}</td></tr>`;
  }
}
bootstrap();
