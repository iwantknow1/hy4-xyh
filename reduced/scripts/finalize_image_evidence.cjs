const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert');
const root='/data/x00951564/codex-hy4-20261027-oldhost-revalidation';
const logs=path.join(root,'logs/resume-exclusive'),out=path.join(root,'model_openbox/hy4');
const read=p=>JSON.parse(fs.readFileSync(p));
const image=read(path.join(logs,'image-inspect.json'))[0];
const container=read(path.join(logs,'image-final-container.json'))[0];
const preflight=read(path.join(logs,'image-final-runtime-preflight.json'));
const report=read(path.join(logs,'image-hccl-6l-graph.json'));
assert.strictEqual(report.status,'passed');assert(Object.values(report.checks).every(x=>x===true));
assert(Object.values(report.endpoint_checks).every(x=>x===true));
for(const endpoint of ['/v1/completions','/v1/chat/completions']){
  const before=report.baseline_by_endpoint[endpoint],after=report.post_update_by_endpoint[endpoint];
  assert.deepStrictEqual(before,after);assert.strictEqual(before.length,2);
  for(const item of after){assert(item.tokens.length>0);assert.strictEqual(item.tokens.length,item.token_logprobs.length);assert(item.token_logprobs.every(Number.isFinite));}
}
assert.strictEqual(report.checkpoint.tensor_count,810);assert.strictEqual(report.inference_world_size,2);
assert.strictEqual(preflight.status,'passed');assert.strictEqual(container.Image,image.Id);
assert.strictEqual(image.Config.Labels['hy4.source.manifest.sha256'],preflight.source_manifest_sha256);
assert.strictEqual(container.Mounts.length,2);assert(container.Mounts.every(x=>['/models','/usr/local/Ascend/driver'].includes(x.Destination)&&x.RW===false));
const serverLog=fs.readFileSync(path.join(logs,'image-hccl-6l-graph-serve.log'),'utf8');
assert(serverLog.includes('Graph capturing finished')&&serverLog.includes('4/4'));
assert(fs.readFileSync(path.join(logs,'image-unit.log'),'utf8').includes('98 passed'));
const excel=read(path.join(root,'performance/hardware_efficiency_fallback.json'));
assert.strictEqual(excel.status,'passed');assert.strictEqual(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'performance/hardware_efficiency.xlsx'))).digest('hex'),excel.workbook_sha256);
const metadata={status:'passed',created_utc:new Date().toISOString(),image_tag:'hy4-codex:a5-cann9.1-20261027-verified',image_id:image.Id,image_uncompressed_bytes:image.Size,base_image_id:'sha256:1ba11fc501306b7ef44b2dcbb7e1fea69a33bc18ca0a7a48c1ede76874a3d6cf',source_manifest_sha256:preflight.source_manifest_sha256,source_file_count:Object.keys(preflight.source_files).length,runtime_versions:preflight.versions,source_mounts:false,external_mounts:container.Mounts,physical_inference_devices:[6,7],physical_producer_device:5,npu_validation:{model:'/models/Hy4-6L',tensor_count:810,tp:2,graph_mode:'FULL_DECODE_ONLY',capture_sizes:[1,2,4,8],weight_transfer:'packed HCCL with explicit CPU staging',checkpoint_manifest_sha256:report.checkpoint.manifest_sha256,endpoint_checks:report.endpoint_checks,all_text_tokens_logprobs_exact:true,evidence:'logs/resume-exclusive/image-hccl-6l-graph.json'},cpu_units:{passed:98,warnings:15,scope:'same frozen production source and runtime in the parent candidate; final overlay only changes reports/Excel/export helper/probe',torch_device_backend_autoload:'0 for CPU mock fixture tests only; NPU inference uses normal backend loading',evidence:'logs/resume-exclusive/image-unit.log'},excel:{status:'passed',sha256:excel.workbook_sha256,formula_recalculation:true},archive:{path:'/mnt/share/x00951564/codex-transfer-20261027/hy4-codex-a5-cann9.1-20261027-verified.tar.gz',status:'exporting'},cards_released:true};
fs.writeFileSync(path.join(out,'IMAGE-METADATA.json'),JSON.stringify(metadata,null,2)+'\n');
let md='# 镜像自包含与实机复验\n\n';
md+='- 镜像：`'+metadata.image_tag+'`\n- Image ID：`'+metadata.image_id+'`\n- 未压缩大小：'+metadata.image_uncompressed_bytes+' 字节\n- 源码身份：`'+metadata.source_manifest_sha256+'`（9 个源文件校验通过）\n\n';
md+='推理容器没有源码或 .venv 挂载，仅只读挂载模型目录和宿主驱动。使用镜像内置源码、uv 环境及 Triton 3.6.0，6L TP2 FULL_DECODE_ONLY 捕获 [1,2,4,8]，完整更新 810 个张量；两个接口、各两条 prompt 的 text/token/logprob 与更新前完全一致。\n\n';
md+='相同冻结生产源码的镜像内 98 项 CPU 单元测试通过。CPU mock 测试单独设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0；实际 NPU 服务不设置该开关。包元数据 torch=2.10.0、运行时 torch.__version__=2.10.0+cpu，校验脚本已区分两者。最初的校验/fixture 失败日志保留。最终派生层仅补充 Excel、报告、公开 JS 导出依赖及校验脚本，没有修改模型/权重传输生产代码。\n\n';
md+='实机验证完成后自有推理和 producer 容器均已停止，NPU 进程表确认 5–7 卡释放。6、7 卡的历史 Warning 状态未擅自复位。\n\n';
md+='导出位置和最终 SHA256 见 IMAGE-METADATA.json、IMAGE-ARCHIVE.sha256；归档使用 docker save + gzip，包含完整镜像层，模型权重不打包。加载：`docker load -i '+metadata.archive.path+'`。启动参数见 IMAGE-README.zh-CN.md。\n';
fs.writeFileSync(path.join(out,'IMAGE-VALIDATION.md'),md);
for(const name of ['VERIFICATION-REPORT.zh-CN.md','report.md','README.md']){
  const p=path.join(out,name);let text=fs.readFileSync(p,'utf8');
  if(!text.includes('镜像最终实机复验'))text+='\n\n## 镜像最终实机复验\n\n[镜像自包含验证与完整 ID](IMAGE-VALIDATION.md)：6L TP2 HCCL 图模式，在没有源码挂载的最终镜像中完整复验通过。5–7 卡已释放。镜像元数据和归档 SHA256 另行交付，避免镜像自身 ID 的循环依赖。\n';
  fs.writeFileSync(p,text);
}
const manifest=read(path.join(out,'manifest.json'));manifest.image_validation=metadata;manifest.verification_matrix.push({scenario:'最终镜像，无源码挂载，6L TP2 HCCL 图模式',status:'passed',evidence:'image-hccl-6l-graph.json',note:'810 张量；双接口 text/token/logprob 严格一致；卡已释放'});fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({image:metadata.image_tag,id:metadata.image_id,npu_validation:'passed',cards_released:true}));
