/** Export the validated single-card analysis with auditable Excel formulas. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [inputPath, outputPath, moduleDirectory, previewDirectory] = process.argv.slice(2);
if (!inputPath || !outputPath) throw new Error('Usage: node export_hardware_efficiency.mjs report.json output.xlsx [node_modules] [preview-dir]');
const moduleUrl = moduleDirectory
  ? pathToFileURL(createRequire(path.join(path.resolve(moduleDirectory), '..', 'package.json')).resolve('@oai/artifact-tool')).href
  : '@oai/artifact-tool';
const { Workbook, SpreadsheetFile } = await import(moduleUrl);
const report = JSON.parse(await fs.readFile(inputPath, 'utf8'));
const workbook = Workbook.create();
const summary = workbook.worksheets.add('Summary');
const kernels = workbook.worksheets.add('Kernels');
const operators = workbook.worksheets.add('Operators');
const exclusions = workbook.worksheets.add('Excluded');
const settings = report.parameters;
const lastKernelRow = report.kernels.length + 4;
if (lastKernelRow > 1048576) throw new Error('Kernel rows exceed the Excel sheet limit');

function literal(value) {
  // Source names can contain Excel formula prefixes. Keep them literal.
  return typeof value === 'string' && /^[=+@-]/.test(value) ? `'${value}` : value;
}
function writeRows(sheet, row, rows) {
  if (rows.length) sheet.getRangeByIndexes(row - 1, 0, rows.length, rows[0].length).values = rows.map(values => values.map(literal));
}
function styleSheet(sheet, title, headers, count) {
  const lastColumn = columnLetter(headers.length);
  const end = Math.max(5, count + 4);
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}${end}`).format.font = { name: 'Arial', size: 11, color: '#172B3A' };
  sheet.getRange('A1').values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = { font: { bold: true, color: '#123C50', size: 16 }, rowHeight: 32 };
  writeRows(sheet, 4, [headers]);
  sheet.getRange(`A4:${lastColumn}4`).format = { fill: '#DCEAF0', font: { bold: true }, wrapText: true, rowHeight: 30, horizontalAlignment: 'center' };
  sheet.getRange(`A5:${lastColumn}${end}`).format.rowHeight = 24;
  sheet.getRange(`A1:${lastColumn}${end}`).format.columnWidth = 18;
  sheet.getRange(`A1:${lastColumn}${end}`).format.verticalAlignment = 'center';
  sheet.getRange(`A5:A${end}`).format.columnWidth = 12;
  sheet.freezePanes.freezeRows(4);
  sheet.tabColor = '#123C50';
}
function columnLetter(index) {
  let result = '';
  for (; index; index = Math.floor((index - 1) / 26)) result = String.fromCharCode(65 + (index - 1) % 26) + result;
  return result;
}

styleSheet(summary, '单卡计算算子加权极致性能达成率', ['指标 / 参数', '值', '说明'], 21);
writeRows(summary, 5, [
  ['加权达成率', null, '无量纲，允许超过 1；不裁剪'],
  ['长算子贡献', null, 'duration ≥ 分界值：Σ(指标 / 经验上限 × 耗时占比)'],
  ['短算子贡献', null, 'duration < 分界值：Σ(耗时占比)'],
  ['计算算子总耗时 (μs)', null, '各次调用耗时之和，包含采集窗口内全部层'],
  ['计算调用数', report.kernels.length, '每次 kernel 调用单独应用分界规则'],
  ['重点优化调用数', null, 'duration > 时长阈值 且 max_ratio < 比例阈值'],
  ['默认经验上限', settings.default_ceiling, '未配置算子规则时使用'],
  ['长短算子分界 (μs)', settings.short_kernel_us, '10 μs 本身属于长算子'],
  ['重点优化时长 (μs)', settings.priority_duration_us, '严格大于，默认 20 μs'],
  ['重点优化比例', settings.priority_ratio, '严格小于，默认 0.6'],
  ['设备', report.device, report.device_selection],
  ['验证范围', report.validation_scope, report.validation_scope === 'functional_smoke' ? '减层随机权重，仅供功能路径验证' : '真实全层权重，层数需与模型审计核对'],
  ['源 CSV', report.source_csv, '原始输入；Kernels 保留源行号和各 ratio'],
  ['源 SHA256', report.source_sha256, '用于追溯采集数据'],
  ['输入 ratio 单位', report.ratio_unit, '明细已统一为 0–1 比例'],
  ['排除行数', report.excluded.length, '见 Excluded：其他卡、通信、拷贝及非计算任务'],
  ['计算口径', report.note, ''],
  ['经验值来源', '默认 0.8；具体算子按配置覆盖', '具体来源见 Kernels 的经验值来源列'],
]);
summary.getRange('B5:B8').formulas = [
  ['=B6+B7'], [`=SUM(Kernels!K5:K${lastKernelRow})`],
  [`=SUM(Kernels!L5:L${lastKernelRow})`], [`=SUM(Kernels!D5:D${lastKernelRow})`],
];
summary.getRange('B10').formulas = [[`=COUNTIF(Kernels!N5:N${lastKernelRow},"重点优化")`]];
summary.getRange('B5:B7').setNumberFormat('0.00%');
summary.getRange('B11').setNumberFormat('0.00%');
summary.getRange('B14').setNumberFormat('0.00%');
summary.getRange('A1:A25').format.columnWidth = 28;
summary.getRange('B1:B25').format.columnWidth = 44;
summary.getRange('C1:C25').format.columnWidth = 72;
summary.getRange('B17:C22').format.wrapText = true;
summary.getRange('A17:C22').format.rowHeight = 48;
summary.getRange('B5').format = { fill: '#DCEAF0', font: { bold: true, size: 16 }, rowHeight: 32 };
summary.getRange('B11:B14').format.fill = '#FFF2CC';
summary.freezePanes.unfreeze();

const ratioHeaders = report.ratio_columns;
const headers = ['CSV 行', '算子名称', '任务类型', 'duration (μs)', 'max_ratio', '选用指标', '选用 ratio', '经验上限', '耗时占比 w', '达成率 x', '长算子贡献', '短算子贡献', '总贡献 w×x', '重点优化', '经验值来源', ...ratioHeaders];
styleSheet(kernels, 'Kernel 调用明细：全层累计，逐次计算', headers, report.kernels.length);
const lastRatioColumn = columnLetter(headers.length);
writeRows(kernels, 5, report.kernels.map(kernel => [
  kernel.row, kernel.operator, kernel.task_type, kernel.duration_us, null,
  kernel.selected_metric, kernel.selected_ratio, kernel.ceiling, null, null, null, null, null, null,
  kernel.ceiling_source, ...ratioHeaders.map(metric => kernel.ratios[metric] ?? null),
]));
const maxFormulas = [], metricFormulas = [], calculationFormulas = [];
let defaultStart = null;
for (let index = 0; index < report.kernels.length; index++) {
  const row = index + 5;
  const kernel = report.kernels[index];
  const metricColumn = columnLetter(16 + ratioHeaders.indexOf(kernel.selected_metric));
  maxFormulas.push([`=MAX(P${row}:${lastRatioColumn}${row})`]);
  metricFormulas.push([`=${metricColumn}${row}`]);
  if (kernel.ceiling_source === '用户约定默认经验值') {
    if (defaultStart === null) defaultStart = row;
  } else if (defaultStart !== null) {
    kernels.getRange(`H${defaultStart}:H${row - 1}`).formulas = Array.from({ length: row - defaultStart }, () => ['=Summary!$B$11']);
    defaultStart = null;
  }
  calculationFormulas.push([
    `=D${row}/Summary!$B$8`, `=G${row}/H${row}`,
    `=IF(D${row}>=Summary!$B$12,I${row}*J${row},0)`,
    `=IF(D${row}<Summary!$B$12,I${row},0)`, `=K${row}+L${row}`,
    `=IF(AND(D${row}>Summary!$B$13,E${row}<Summary!$B$14),"重点优化","")`,
  ]);
}
if (defaultStart !== null) kernels.getRange(`H${defaultStart}:H${lastKernelRow}`).formulas = Array.from({ length: lastKernelRow - defaultStart + 1 }, () => ['=Summary!$B$11']);
kernels.getRange(`E5:E${lastKernelRow}`).formulas = maxFormulas;
kernels.getRange(`G5:G${lastKernelRow}`).formulas = metricFormulas;
kernels.getRange(`I5:N${lastKernelRow}`).formulas = calculationFormulas;
kernels.getRange(`E5:M${lastKernelRow}`).setNumberFormat('0.00%');
kernels.getRange(`F5:F${lastKernelRow}`).setNumberFormat('@');
kernels.getRange(`P5:${lastRatioColumn}${lastKernelRow}`).setNumberFormat('0.00%');
kernels.getRange(`D5:D${lastKernelRow}`).setNumberFormat('0.00');
kernels.getRange(`B1:B${lastKernelRow}`).format.columnWidth = 40;
kernels.getRange(`O1:O${lastKernelRow}`).format.columnWidth = 45;
kernels.getRange(`F1:F${lastKernelRow}`).format.columnWidth = 24;
kernels.getRange(`N5:N${lastKernelRow}`).conditionalFormats.addCustom('N5="重点优化"', { fill: '#FEE2E2', font: { bold: true, color: '#991B1B' } });
kernels.freezePanes.freezeColumns(2);

styleSheet(operators, '算子汇总：包含各层和各步的全部调用', ['算子名称', '调用数', '累计耗时 (μs)', '耗时占比', '加权贡献', '重点调用数'], report.operator_summary.length);
writeRows(operators, 5, report.operator_summary.map(group => [group.operator, null, null, null, null, null]));
const operatorFormulas = [];
for (let index = 0; index < report.operator_summary.length; index++) {
  const row = index + 5;
  // Escape Excel criteria wildcards so operator names are matched literally.
  const criterion = `SUBSTITUTE(SUBSTITUTE(SUBSTITUTE(A${row},"~","~~"),"*","~*"),"?","~?")`;
  operatorFormulas.push([
    `=COUNTIF(Kernels!$B$5:$B$${lastKernelRow},${criterion})`,
    ...['D', 'I', 'M'].map(column => `=SUMIF(Kernels!$B$5:$B$${lastKernelRow},${criterion},Kernels!$${column}$5:$${column}$${lastKernelRow})`),
    `=COUNTIFS(Kernels!$B$5:$B$${lastKernelRow},${criterion},Kernels!$N$5:$N$${lastKernelRow},"重点优化")`,
  ]);
}
operators.getRange(`B5:F${report.operator_summary.length + 4}`).formulas = operatorFormulas;
operators.getRange(`A1:A${report.operator_summary.length + 4}`).format.columnWidth = 44;
operators.getRange(`D5:E${report.operator_summary.length + 4}`).setNumberFormat('0.00%');

styleSheet(exclusions, '排除记录：保留原始行便于复核筛选范围', ['CSV 行', '排除原因', '源数据 (JSON)'], report.excluded.length);
writeRows(exclusions, 5, report.excluded.length ? report.excluded.map(row => [row.row, row.reason, JSON.stringify(row.source)]) : [[null, '无排除行', null]]);
exclusions.getRange('B1:B25').format.columnWidth = 38;
exclusions.getRange('C1:C25').format.columnWidth = 100;
if (report.excluded.length) {
  const sourceRange = exclusions.getRange(`C5:C${report.excluded.length + 4}`);
  sourceRange.format.wrapText = true;
  sourceRange.format.autofitRows();
}

workbook.recalculate();
const score = Number(summary.getRange('B5').values[0][0]);
if (!Number.isFinite(score) || Math.abs(score - report.weighted_attainment) > 1e-9) throw new Error(`Excel/Python calculation mismatch: ${score} vs ${report.weighted_attainment}`);
for (let index = 0; index < report.operator_summary.length; index++) {
  const actual = operators.getRange(`B${index + 5}:F${index + 5}`).values[0].map(Number);
  const group = report.operator_summary[index];
  const expected = [group.calls, group.duration_us, group.dur_ratio, group.contribution, group.priority_calls];
  if (actual.some((value, column) => !Number.isFinite(value) || Math.abs(value - expected[column]) > 1e-8)) throw new Error(`Operator aggregation mismatch: ${group.operator}`);
}
console.log((await workbook.inspect({ kind: 'region', sheetId: 'Summary', range: 'A5:B14', maxChars: 1500, tableMaxRows: 10, tableMaxCols: 2 })).ndjson);
const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 20 }, maxChars: 1500 });
console.log(errors.ndjson);
if (previewDirectory) {
  await fs.mkdir(previewDirectory, { recursive: true });
  for (const [name, range] of [['Summary', 'A1:C22'], ['Kernels', 'A1:N12'], ['Operators', 'A1:F14'], ['Excluded', 'A1:C10']]) {
    const preview = await workbook.render({ sheetName: name, range, scale: 1.2, format: 'png' });
    await fs.writeFile(path.join(previewDirectory, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
