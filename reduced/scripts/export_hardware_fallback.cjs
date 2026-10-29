const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert');
const XLSX = require('../tools/node_modules/xlsx/dist/xlsx.full.min.js');
const calculate = require('../tools/node_modules/xlsx-calc');

const [input, output] = process.argv.slice(2);
assert(input && output, 'Usage: node export_hardware_fallback.cjs input.json output.xlsx');
const data = JSON.parse(fs.readFileSync(input));
const wb = XLSX.utils.book_new();
const end = data.kernels.length + 1;
const parameters = data.parameters;
const summaryRows = [
  ['HY4 reduced-model single-device compute estimate', 'Value'],
  ['validation_scope', data.validation_scope],
  ['physical_device', String(data.device)],
  ['default_ceiling (editable)', parameters.default_ceiling],
  ['short_kernel_us (editable)', parameters.short_kernel_us],
  ['priority_duration_us (editable)', parameters.priority_duration_us],
  ['priority_ratio (editable)', parameters.priority_ratio],
  ['compute_duration_us', null],
  ['weighted_attainment', null],
  ['priority_calls', null],
  ['long_contribution', null],
  ['short_contribution', null],
  ['compute_calls', data.kernels.length],
  ['excluded_calls', data.excluded.length],
  ['source_csv_sha256', data.source_sha256],
  ['scope', 'All two decoder layers, ten worker graph executions; one physical device.'],
  ['interpretation', 'Compute-call duration weighting; excludes communication/idle/overlap. Not end-to-end speedup.'],
  ['ceiling_source', 'Generic default 0.8; no shape/chip-specific expert ceiling supplied.'],
  ['original_exporter', '@oai/artifact-tool unavailable; original failure retained in input JSON.'],
  ['fallback', 'SheetJS 0.18.5 + xlsx-calc 0.9.2; formulas independently recalculated and checked.'],
];
const summary = XLSX.utils.aoa_to_sheet(summaryRows);
function formula(sheet, address, value) { sheet[address] = {t: 'n', f: value}; }
formula(summary, 'B8', `SUM(Kernels!D2:D${end})`);
formula(summary, 'B9', `SUM(Kernels!J2:J${end})`);
formula(summary, 'B10', `SUM(Kernels!K2:K${end})`);
formula(summary, 'B11', `SUM(Kernels!L2:L${end})`);
formula(summary, 'B12', `SUM(Kernels!M2:M${end})`);
XLSX.utils.book_append_sheet(wb, summary, 'Summary');

const rawColumns = Object.keys(data.kernels[0].source);
const headers = ['CSV row', 'Operator', 'Task type', 'Duration us', 'Max ratio',
  'Selected ratio', 'Ceiling', 'Attainment', 'Duration weight', 'Contribution',
  'Priority', 'Long contribution', 'Short contribution', 'Selected metric',
  'Ceiling source', ...data.ratio_columns, ...rawColumns.map(x => `Raw: ${x}`)];
const rows = [headers];
const groups = new Map();
for (let i = 0; i < data.kernels.length; i++) {
  const k = data.kernels[i], row = i + 2;
  const usesDefault = k.ceiling_source === '用户约定默认经验值';
  rows.push([k.row, k.operator, k.task_type, k.duration_us, k.max_ratio,
    k.selected_ratio, usesDefault ? null : k.ceiling, null, null, null, null,
    null, null, k.selected_metric, k.ceiling_source,
    ...data.ratio_columns.map(metric => k.ratios[metric] ?? null),
    ...rawColumns.map(column => k.source[column])]);
  if (!groups.has(k.operator)) groups.set(k.operator, []);
  groups.get(k.operator).push(row);
}
const kernels = XLSX.utils.aoa_to_sheet(rows);
for (let i = 0; i < data.kernels.length; i++) {
  const k = data.kernels[i], r = i + 2;
  if (k.ceiling_source === '用户约定默认经验值') formula(kernels, `G${r}`, 'Summary!$B$4');
  formula(kernels, `H${r}`, `F${r}/G${r}`);
  formula(kernels, `I${r}`, `D${r}/Summary!$B$8`);
  formula(kernels, `J${r}`, `IF(D${r}<Summary!$B$5,I${r},I${r}*H${r})`);
  formula(kernels, `K${r}`, `IF(AND(D${r}>Summary!$B$6,E${r}<Summary!$B$7),1,0)`);
  formula(kernels, `L${r}`, `IF(D${r}>=Summary!$B$5,J${r},0)`);
  formula(kernels, `M${r}`, `IF(D${r}<Summary!$B$5,J${r},0)`);
}
XLSX.utils.book_append_sheet(wb, kernels, 'Kernels');
const operatorRows = [['Operator', 'Calls', 'Duration us', 'Duration weight', 'Contribution', 'Priority calls']];
const orderedGroups = [...groups].sort((a, b) => {
  const total = rows => rows.reduce((s, r) => s + data.kernels[r - 2].duration_us, 0);
  return total(b[1]) - total(a[1]);
});
for (const [name, indices] of orderedGroups) operatorRows.push([name, indices.length]);
const operators = XLSX.utils.aoa_to_sheet(operatorRows);
orderedGroups.forEach(([, indices], i) => {
  for (const [column, source] of [['C', 'D'], ['D', 'I'], ['E', 'J'], ['F', 'K']]) {
    formula(operators, `${column}${i + 2}`, `SUM(${indices.map(r => `Kernels!${source}${r}`).join(',')})`);
  }
});
XLSX.utils.book_append_sheet(wb, operators, 'Operators');
const excludedRows = [['CSV row', 'Exclusion reason', ...rawColumns],
  ...data.excluded.map(k => [k.row, k.reason, ...rawColumns.map(c => k.source[c])])];
XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(excludedRows), 'Excluded');

for (const sheet of Object.values(wb.Sheets)) {
  sheet['!autofilter'] = {ref: sheet['!ref']};
  sheet['!cols'] = [{wch: 18}, {wch: 65}, ...Array(80).fill({wch: 18})];
}
summary['!cols'] = [{wch: 48}, {wch: 110}];
wb.Workbook = {CalcPr: {calcMode: 'auto', fullCalcOnLoad: true}};
function close(actual, expected, label) {
  assert(Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
}
calculate(wb);
for (let i = 0; i < data.kernels.length; i++) {
  const r = i + 2, k = data.kernels[i];
  close(kernels[`H${r}`].v, k.attainment, `attainment ${r}`);
  close(kernels[`I${r}`].v, k.dur_ratio, `weight ${r}`);
  close(kernels[`J${r}`].v, k.contribution, `contribution ${r}`);
  assert.strictEqual(kernels[`K${r}`].v, Number(k.priority), `priority ${r}`);
}
close(summary.B8.v, data.compute_duration_us, 'total duration');
close(summary.B9.v, data.weighted_attainment, 'weighted attainment');
close(summary.B10.v, data.priority_calls, 'priority count');
close(summary.B11.v, data.long_contribution, 'long contribution');
close(summary.B12.v, data.short_contribution, 'short contribution');

summary.B4.v = 0.9;
summary.B5.v = 12;
calculate(wb);
const changedExpected = data.kernels.reduce((sum, k) => {
  const ceiling = k.ceiling_source === '用户约定默认经验值' ? 0.9 : k.ceiling;
  return sum + k.duration_us / data.compute_duration_us * (k.duration_us < 12 ? 1 : k.selected_ratio / ceiling);
}, 0);
close(summary.B9.v, changedExpected, 'editable parameter recalculation');
const parameterChangeResult = summary.B9.v;
summary.B4.v = parameters.default_ceiling;
summary.B5.v = parameters.short_kernel_us;
calculate(wb);
close(summary.B9.v, data.weighted_attainment, 'restored defaults');

fs.writeFileSync(output, XLSX.write(wb, {bookType: 'xlsx', type: 'buffer', compression: true}));
const reopened = XLSX.read(fs.readFileSync(output), {type: 'buffer', cellFormula: true});
assert.deepStrictEqual(reopened.SheetNames, ['Summary', 'Kernels', 'Operators', 'Excluded']);
close(reopened.Sheets.Summary.B9.v, data.weighted_attainment, 'reopened workbook cache');
calculate(reopened);
close(reopened.Sheets.Summary.B9.v, data.weighted_attainment, 'reopened workbook recalculation');
const report = {
  feature: 'hardware_efficiency_excel', status: 'passed', validation_scope: data.validation_scope,
  original_exporter: 'failed: @oai/artifact-tool unavailable; original JSON and logs retained',
  fallback_libraries: {xlsx: XLSX.version, 'xlsx-calc': require('../tools/node_modules/xlsx-calc/package.json').version},
  workbook: '/codex/performance/hardware_efficiency.xlsx',
  workbook_sha256: crypto.createHash('sha256').update(fs.readFileSync(output)).digest('hex'),
  source_csv_sha256: data.source_sha256, physical_device: data.device,
  compute_calls: data.kernels.length, excluded_calls: data.excluded.length,
  weighted_attainment: reopened.Sheets.Summary.B9.v,
  checks: {all_kernel_formulas_match_json: true, summary_formulas_match_json: true,
    editable_parameter_recalculation: true, restored_defaults_match: true,
    saved_workbook_reopened_and_recalculated: true},
  parameter_change_probe: {ceiling: 0.9, short_kernel_us: 12, weighted_attainment: parameterChangeResult},
  note: data.note,
};
fs.writeFileSync(path.join(path.dirname(output), 'hardware_efficiency_fallback.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
