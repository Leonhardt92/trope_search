import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0/+esm';

env.allowLocalModels = false;

const MODEL_ID = 'Xenova/bge-small-zh-v1.5';
const COLUMN_DEFS = [
  { key: 'id', label: 'id' },
  { key: 'title', label: '标题' },
  { key: 'template', label: '空白模板' },
  { key: 'relationship', label: '角色关系' },
  { key: 'cast_size', label: '人数' },
  { key: 'logline', label: '一句话定义' },
  { key: 'location', label: '地点' },
  { key: 'forbidden_elements', label: '禁止使用元素' },
  { key: 'self_check', label: '自检问题' },
  { key: 'examples', label: '示例' },
];

const COLUMNS = COLUMN_DEFS.map(({ key }) => key);
const LABELS = Object.fromEntries(COLUMN_DEFS.map(({ key, label }) => [key, label]));
const FULL_WIDTH_COLUMNS = new Set(['template', 'logline', 'forbidden_elements', 'self_check', 'examples']);
const TEXTAREA_COLUMNS = new Set(['template', 'logline', 'forbidden_elements', 'self_check', 'examples']);
const EXCLUDED_EMBEDDING_COLUMNS = new Set(['id', 'forbidden_elements', 'self_check', 'examples']);
const DEFAULT_EMBEDDING_COLUMNS = new Set(['title', 'template', 'relationship', 'cast_size', 'logline', 'location']);

let extractor = null;

const els = {
  rawLine: document.getElementById('rawLine'),
  parseBtn: document.getElementById('parseBtn'),
  fillExampleBtn: document.getElementById('fillExampleBtn'),
  fieldGrid: document.getElementById('fieldGrid'),
  checkboxGrid: document.getElementById('checkboxGrid'),
  generateBtn: document.getElementById('generateBtn'),
  copyDataBtn: document.getElementById('copyDataBtn'),
  copyEmbBtn: document.getElementById('copyEmbBtn'),
  dataOutput: document.getElementById('dataOutput'),
  embOutput: document.getElementById('embOutput'),
  status: document.getElementById('status'),
};

const fieldInputs = new Map();
const checkboxes = new Map();

const EXAMPLE_LINE = '2,电梯里的临时同盟,几名原本互不信任的人被困在____，必须在____之前达成合作。,临时盟友|互相提防,3-5人,用封闭空间强迫人物快速暴露真实立场。,写字楼电梯,血浆|怪物露正脸|长篇回忆杀,每个人在电梯里有没有独立的利益冲突？,https://www.youtube.com/watch?v=3JZ_D3ELwOQ';

function setStatus(text, type = 'normal') {
  els.status.textContent = text;
  els.status.className = 'status' + (type === 'ok' ? ' ok' : type === 'error' ? ' error' : '');
}

function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    const next = line[i + 1];
    if (ch === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

function escapeCSV(value) {
  const s = String(value ?? '');
  if (/[",\n]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function renderFields() {
  els.fieldGrid.innerHTML = '';
  for (const { key, label } of COLUMN_DEFS) {
    const wrap = document.createElement('div');
    wrap.className = 'field' + (FULL_WIDTH_COLUMNS.has(key) ? ' full' : '');
    const fieldLabel = document.createElement('label');
    fieldLabel.textContent = label;
    const input = document.createElement(TEXTAREA_COLUMNS.has(key) ? 'textarea' : 'input');
    if (input.tagName === 'INPUT') input.type = 'text';
    input.value = '';
    input.rows = 3;
    wrap.appendChild(fieldLabel);
    wrap.appendChild(input);
    els.fieldGrid.appendChild(wrap);
    fieldInputs.set(key, input);
  }
}

function renderCheckboxes() {
  els.checkboxGrid.innerHTML = '';
  for (const { key, label } of COLUMN_DEFS) {
    if (EXCLUDED_EMBEDDING_COLUMNS.has(key)) continue;
    const wrap = document.createElement('label');
    wrap.className = 'check-item';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = DEFAULT_EMBEDDING_COLUMNS.has(key);
    checkboxes.set(key, cb);
    wrap.appendChild(cb);
    wrap.appendChild(document.createTextNode(label));
    els.checkboxGrid.appendChild(wrap);
  }
}

function setFieldsFromLine(line) {
  const parts = parseCSVLine(line.trim());
  if (parts.length !== COLUMNS.length) {
    throw new Error(`列数不对。当前解析到 ${parts.length} 列，预期 ${COLUMNS.length} 列。`);
  }
  COLUMNS.forEach((col, idx) => {
    fieldInputs.get(col).value = parts[idx] ?? '';
  });
}

function getCurrentRecord() {
  const row = {};
  for (const col of COLUMNS) {
    row[col] = fieldInputs.get(col).value.trim();
  }
  return row;
}

function buildDataCSV(row) {
  const header = COLUMNS.join(',');
  const data = COLUMNS.map(col => escapeCSV(row[col])).join(',');
  return header + '\n' + data;
}

async function initModel() {
  if (extractor) return extractor;
  setStatus('正在初始化模型，首次下载会比较慢……当前版本默认使用 WASM，避免部分 Mac/WebGPU 环境报错。');
  try {
    extractor = await pipeline('feature-extraction', MODEL_ID, {
      dtype: 'q8',
    });
    setStatus('模型初始化完成。当前使用 WASM（更稳）。', 'ok');
    return extractor;
  } catch (err) {
    const detail = err?.message || String(err);
    throw new Error(`模型初始化失败：${detail}`);
  }
}

async function embedText(text) {
  const clean = String(text ?? '').trim();
  if (!clean) return '';
  const out = await extractor(clean, { pooling: 'mean', normalize: true });
  const vector = typeof out.tolist === 'function' ? out.tolist()[0] : Array.from(out.data || []);
  return JSON.stringify(vector);
}

async function buildEmbeddingCSV(row) {
  const selected = [];
  for (const [col, cb] of checkboxes.entries()) {
    if (cb.checked) selected.push(col);
  }
  if (!selected.length) throw new Error('至少要勾选一列用于 embedding。');

  const header = ['id', ...selected.map(col => `${col}__embedding`)];
  const values = [escapeCSV(row.id)];

  for (const col of selected) {
    setStatus(`正在生成 embedding：${LABELS[col]}`);
    const vec = await embedText(row[col]);
    values.push(escapeCSV(vec));
  }

  return header.join(',') + '\n' + values.join(',');
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
}

renderFields();
renderCheckboxes();
els.rawLine.value = EXAMPLE_LINE;
try { setFieldsFromLine(EXAMPLE_LINE); } catch {}

els.fillExampleBtn.addEventListener('click', () => {
  els.rawLine.value = EXAMPLE_LINE;
  try {
    setFieldsFromLine(EXAMPLE_LINE);
    setStatus('示例已填入。', 'ok');
  } catch (err) {
    setStatus(err.message, 'error');
  }
});

els.parseBtn.addEventListener('click', () => {
  try {
    const line = els.rawLine.value.trim();
    if (!line) throw new Error('请先输入一行 CSV。');
    setFieldsFromLine(line);
    setStatus('解析完成，你现在可以直接修改每个字段。', 'ok');
  } catch (err) {
    setStatus(err.message, 'error');
  }
});

els.generateBtn.addEventListener('click', async () => {
  try {
    const row = getCurrentRecord();
    if (!row.id) throw new Error('id 不能为空。');
    els.dataOutput.value = buildDataCSV(row);
    els.embOutput.value = '';
    setStatus('data.csv 已生成，正在生成 embeddings.csv ……');
    await initModel();
    els.embOutput.value = await buildEmbeddingCSV(row);
    setStatus('两个 CSV 记录已生成完成。', 'ok');
  } catch (err) {
    console.error(err);
    const detail = err?.stack ? `${err.message || String(err)}\n\n${err.stack}` : (err?.message || String(err));
    setStatus(detail, 'error');
  }
});

els.copyDataBtn.addEventListener('click', async () => {
  try {
    if (!els.dataOutput.value.trim()) throw new Error('还没有生成 data.csv 输出。');
    await copyText(els.dataOutput.value);
    setStatus('已复制 data.csv 输出。', 'ok');
  } catch (err) {
    setStatus(err.message, 'error');
  }
});

els.copyEmbBtn.addEventListener('click', async () => {
  try {
    if (!els.embOutput.value.trim()) throw new Error('还没有生成 embeddings.csv 输出。');
    await copyText(els.embOutput.value);
    setStatus('已复制 embeddings.csv 输出。', 'ok');
  } catch (err) {
    setStatus(err.message, 'error');
  }
});
