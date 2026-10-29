# 功能与 AISBench 验证

基础服务先用本地 curl 做可达性和文本完整性检查，再按用户要求调用目标版本的 AISBench。执行前清除会拦截本地请求的代理变量。

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
curl --fail --silent --show-error http://<host>:<port>/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","prompt":"Hello","max_tokens":16,"temperature":0}'
```

基础功能可判定为正常的最低证据是 HTTP 成功、存在 `choices[].text`、无 Traceback/错误字段且文本无明显乱码。该判断不等价于精度通过；精度结论应记录 AISBench 或客户指定基准的真实结果。
