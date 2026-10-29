# SSH 登录服务器排障与执行指南

本文是 `model-rl-case-guard` 的登录与远端执行参考，覆盖 Windows 宿主机、vLLM-Ascend 单机/多机部署以及 SSH 工具的通用排障方法。

相关实现：

- [环境安装 Step 1：服务器连接](../../environment-install/references/step-01-connect.md)
- [vLLM-Ascend 服务连接模式](../../vllm-ascend-serving/SKILL.md)
- [vLLM-Ascend 服务生命周期](../../vllm-ascend-serving/references/behavior.md)
- [SSH 工具实现](../../../common_tools/version-bisect/scripts/ssh_utils.py)
- [多机验证流程](multi-node.md)

## 1. 先确认连接拓扑

在任何远端变更前，先记录完整参数。参数缺失时只做只读探测，不要猜测目标机器、容器或环境。

| 参数 | 说明 |
| --- | --- |
| 目标地址 | `--host` 是宿主机 IP，还是容器/虚拟机自身 IP |
| SSH 端口 | 默认 22；非 22 端口的 known_hosts 键名为 `[host]:port` |
| 用户名 | 默认 `root`，也可能是普通用户 |
| 连接模式 | `docker`、`conda` 或 `direct` |
| 凭据 | 私钥路径优先；密码不得写入脚本、命令行历史或日志 |
| Docker 容器 | `docker` 模式必填，head/worker 可不同 |
| Conda 环境 | `conda` 模式必填，head/worker 可不同 |
| CANN 根目录 | 默认 `/usr/local/Ascend`，执行前确认实际路径 |
| 模型路径、TP/DP | 作为后续服务预检参数一起记录 |
| 多机信息 | head、worker 列表，worker 凭据/容器/环境是否覆盖继承值 |

### 三种模式的语义

| 模式 | SSH 登录目标 | 登录成功后的命令位置 | 典型认证 |
| --- | --- | --- | --- |
| `docker` | 宿主机 | `docker exec` 进入指定容器 | 私钥，或脚本用密码引导生成受控密钥 |
| `conda` | 宿主机 | 激活指定 Conda 环境后在宿主机执行 | 私钥，或脚本用密码引导生成受控密钥 |
| `direct` | 容器/虚拟机自身 | 直接在 SSH 目标内执行 | 密码或显式私钥；不做 Docker/Conda 包装，不自动注入密钥 |

多机时所有节点必须使用同一种模式。worker 默认继承 head 的模式、用户、凭据、容器和 Conda 环境；只有实际不同才使用 `--worker-*` 覆盖。

## 2. 推荐决策顺序

### 2.1 先做只读连通性探测

第一条探测应确认登录的是哪台机器、哪个用户，而不是直接启动服务：

~~~bash
ssh -p <port> \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=10 \
  <user>@<host> "hostname && whoami"
~~~

- `accept-new` 只自动接受新主机指纹；已有指纹发生变化时应停止并核对，不要改成 `StrictHostKeyChecking=no` 绕过。
- 有明确私钥时可加 `-i <key-file> -o BatchMode=yes`。`BatchMode=yes` 用于确认免密是否真的可用，不会弹密码提示。
- 记录主机名和用户名，避免命令执行到错误的机器。

### 2.2 认证通道选择

按以下顺序选择，不要在同一种失败通道上循环重试：

1. **已有且已确认的 SSH 私钥**：显式传 `--key-file`，或用 `ssh -i` 配合 `BatchMode=yes`。
2. **vLLM-Ascend serving 的受控密码引导**：`docker`/`conda` 模式可由服务脚本在本地 `.workspace-local/ssh-keys/` 生成目标主机专用 Ed25519 密钥，并通过一次密码连接追加公钥；后续操作使用该密钥。仅在用户已授权写入目标账号 `~/.ssh/authorized_keys` 时启用，`direct` 模式不会自动注入。
3. **Paramiko 密码登录**：环境安装流程使用 `timeout=30`、`look_for_keys=False`，并设置 transport keepalive 30 秒；适合多条预检或 `sudo -S`。
4. **受控交互式登录**：在沙箱外的普通 PowerShell、WSL 或 Git Bash 终端手动确认密码和主机指纹。
5. **切换工具或环境**：若当前 Windows 沙箱不能创建 askpass 子进程、Git Bash 无法启动或 Paramiko 依赖损坏，直接切换通道并保留首次错误。

服务管理应调用 `vllm-ascend-serving/scripts/`，不要手工拼接带密码、环境变量和 `vllm serve` 的长 SSH 命令。脚本已实现参数转义、密码 askpass、连接模式和生命周期清理。

### 2.3 Windows 密码认证的实现要点

仓库 serving 脚本针对 Windows 使用 ASCII 系统临时目录中的 `.pwd` + `.cmd` askpass 文件，并设置 `SSH_ASKPASS`、`SSH_ASKPASS_REQUIRE=force`、`DISPLAY=dummy`。

这可以规避工作区含中文路径时 `CreateProcessW` 无法启动裸 `.py` askpass，以及密码中特殊字符被 `cmd.exe` 重新解析的问题。不要把密码拼到 `ssh` 命令参数中；优先让脚本管理 askpass 文件并在退出时清理。

## 3. Paramiko 安全连接模板

仓库环境安装 skill 使用 Paramiko 密码连接：`timeout=30`、`auth_timeout=30`、`banner_timeout=30`、`look_for_keys=False`、`allow_agent=False`，并将 transport keepalive 设为 30 秒。生产环境应先加载并核对 known_hosts；仅隔离测试环境才考虑 `AutoAddPolicy`。

~~~python
import paramiko

ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
ssh.connect(
    "<host>", port=22, username="<user>", password="[REDACTED_SECRET]",
    timeout=30, auth_timeout=30, banner_timeout=30,
    look_for_keys=False, allow_agent=False,
)
transport = ssh.get_transport()
if transport is not None:
    transport.set_keepalive(30)
stdin, stdout, stderr = ssh.exec_command("hostname && whoami", timeout=60)
print(stdout.read().decode("utf-8", errors="replace"))
print(stderr.read().decode("utf-8", errors="replace"))
ssh.close()
~~~

### 使用 sudo 的安全方式

不要使用 `echo <password> | sudo`，也不要把密码放到远程命令字符串中。通过 Paramiko 的 stdin 写入密码，避免出现在远程 `ps` 和命令历史中：

~~~python
def sudo_run(ssh, server_password, command, timeout=60):
    safe_command = command.replace("'", "'\\''")
    remote = "sudo -S -p '' bash -c '" + safe_command + "'"
    stdin, stdout, stderr = ssh.exec_command(remote, timeout=timeout)
    stdin.write(server_password + "\n")
    stdin.flush()
    return (
        stdout.read().decode("utf-8", errors="replace"),
        stderr.read().decode("utf-8", errors="replace"),
    )
~~~

`command` 仍需正确 shell 转义，尤其是单引号、`&`、`$` 和双引号。普通用户没有 sudo 权限时，应改用有写权限的工作目录或请用户提供授权，不要假定可以切 root。

### 仓库 SSH 工具的共同约定

`common_tools` 中的 `SSHClient` 支持 `host`、`port`、`username`、密码和 `key_path`，连接超时默认为 30 秒；`exec()` 返回退出码、标准输出和标准错误，统一按 UTF-8（替换非法字节）解码，并提供 SFTP 上传能力。`DockerExecutor` 和 `CondaExecutor` 都复用这套连接，分别负责 `docker exec` 和 Conda 环境包装。因此排查时应分别记录“SSH 主机连接成功”和“容器/Conda 上下文执行成功”，不要把后者失败误判为 SSH 失败。

远端命令可能包含模型路径、环境变量或 JSON 参数时，交给 serving/benchmark 脚本处理引用和转义；不要在外层再次拼接未经转义的 `bash -c` 字符串。

## 4. 登录成功后的最小预检

先执行只读检查，再进入模型开箱或服务启动：

~~~bash
hostname
whoami
pwd
command -v npu-smi || true
command -v vllm || true
python3 --version || python --version
df -h / /home /tmp 2>/dev/null
free -h 2>/dev/null || true
~~~

NPU、CANN 和 vLLM 检查分开执行：

~~~bash
test -f /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
test -f /usr/local/Ascend/nnal/atb/set_env.sh && \
  source /usr/local/Ascend/nnal/atb/set_env.sh
command -v npu-smi && npu-smi info
python3 -c "import torch, torch_npu, vllm; print('imports: OK')"
~~~

- `npu-smi` 可用不等于目标 CANN 根目录已正确配置。
- 出现 `libc_sec.so` 等动态库错误时，先确认 CANN `set_env.sh` 和 `LD_LIBRARY_PATH`，不要立即归因于 SSH 或模型。
- 记录 `which vllm`、Python 解释器、CANN 路径和容器/环境名称，服务启动必须使用同一上下文。

### 按连接模式检查

**docker**

1. 在宿主机执行 `docker ps --format '{{.Names}}'`，确认容器存在且运行中。
2. 通过服务脚本进入容器执行 `npu-smi`、Python 导入和模型路径检查。
3. 容器不需要运行 sshd；不要把容器 IP 当成 `--host`。

**conda**

1. 确认 `conda` 命令可用且环境名存在。
2. 非交互命令使用登录 shell 或显式 source Conda 初始化脚本，再检查 `which python` 和 `which vllm`。
3. CANN 环境变量须在该环境上下文中重新确认。

**direct**

1. 确认目标容器/虚拟机自身运行 sshd，且 IP 可直接登录。
2. 不传 `--container` 或 `--conda-env`，也不依赖宿主机 Docker 权限。
3. 服务脚本不会为 direct 模式自动生成或追加 SSH 公钥。

## 5. 多机登录与开箱前置

多机验证遵循“全部节点预检通过后，才允许任何节点启动”原则：

1. 对 head 和每个 worker 分别执行 `hostname && whoami`，记录实际返回值。
2. 按各节点模式检查容器/Conda 环境、CANN、`npu-smi`、Python 导入和模型路径。
3. 检查 head 的 DP RPC 端口以及节点间网络策略；管理 IP 可达不代表 DP 端口可达。
4. worker 凭据、容器和环境不同则显式设置 `--worker-password`、`--worker-key-file`、`--worker-container` 或 `--worker-conda-env`。
5. 由服务脚本统一启动和清理：head 不得带 `--headless`，只有 worker 使用 engine-only/headless。
6. 任一节点失败时停止后续启动，修复后从全量预检重新开始。

大模型加载和编译可能持续很久，不要把“SSH 连接仍存活但远端暂时无输出”误判为登录失败。脚本默认健康检查约 1800 秒、DP RPC 等待约 3600 秒；长任务使用 keepalive 30 秒，单条命令读取超时可设 60～120 秒。

## 6. Windows / DSH 已知障碍

以下现象是受限 Windows 执行环境中的通用风险信号，遇到后应切换方案，不要重复原通道。

| 方式或症状 | 根因判断 | 对策 |
| --- | --- | --- |
| Windows OpenSSH + SSH_ASKPASS 指向 .sh，报 posix_spawnp: No such file or directory | Windows OpenSSH 无法直接 spawn shell 脚本 | 改用 serving 脚本的 .cmd askpass，或在沙箱外普通终端执行 |
| Windows OpenSSH + .exe askpass，报 CreateProcessW failed error:2 | 受限沙箱拦截 ssh 创建 askpass 子进程；通常不是密码或路径错误 | 不再反复改路径；改用 Paramiko、免密私钥或沙箱外终端 |
| Git Bash 报 CreateFileMapping ... Win32 error 5 | DSH 沙箱限制 Git Bash 的内存映射初始化 | 使用 PowerShell + py -3；必要时在沙箱外用 Git Bash/WSL |
| Paramiko 导入 cryptography._rust 失败，DLL load failed | 缺少 MSVC 运行库或 Python/cryptography 二进制不匹配 | 安装 VC++ Redistributable 后重试；或改用私钥 ssh/plink |
| Permission denied (publickey) | 私钥未提供、权限不对或服务端未授权 | 核对 --key-file、用户名和 authorized_keys；获授权时用密码引导受控密钥 |
| Permission denied (password) | 密码、用户名或认证策略不匹配 | 核对账号；禁止把密码写入命令行；切换到交互式终端 |
| Host key verification failed | known_hosts 指纹不匹配，可能是重装或中间人风险 | 停止操作，核对管理员提供的新指纹后再更新记录 |
| 连接超时 / No route to host | 网络、端口、防火墙或代理问题 | 先验证目标 IP、SSH 端口和路由，不要立即修改服务参数 |
| docker: No such container | 宿主机可登录，但容器名错误或未运行 | 在宿主机只读列出容器，修正 --container |
| conda: command not found 或环境不存在 | 非交互 shell 未初始化 Conda 或环境名错误 | source Conda 初始化脚本，确认 conda env list |
| sudo: a password is required / not in the sudoers file | 普通用户无 sudo 或 sudo 需要 TTY/额外策略 | 使用 stdin 的 sudo -S，或改到可写目录/请求管理员授权 |
| libc_sec.so、libascendcl.so 找不到 | CANN 环境未 source 或路径不匹配 | 确认 --cann-dir，source 对应 set_env.sh，再跑 npu-smi |
| worker 登录成功但预检失败 | worker 模式、凭据、容器/环境或网络端口不正确 | 显式覆盖 --worker-* 参数，单独记录并修复 worker |
| 服务健康检查超时 | 模型加载/编译、端口、CANN 或 vLLM 进程问题，不一定是 SSH | 读取远端 stdout/stderr 和服务状态，区分 SSH、运行时、模型路径和端口 |

## 7. 重试、记录与安全边界

- 同一种认证方式最多尝试 1～2 次；第二次仍失败就切换通道，并保留第一条完整错误。
- 不记录密码、私钥内容、askpass 文件内容或完整带密钥命令；日志只记录主机别名、端口、模式、用户名、成功/失败和错误类别。
- 密码通过 Paramiko stdin 或 serving 脚本的受控 askpass 传递；不使用 sshpass -p、echo password | sudo 或命令行 --password 的日志回显。
- 私钥文件应限制本地权限；非 22 端口要正确匹配 [host]:port 的 known_hosts 条目。
- 不以 StrictHostKeyChecking=no 作为默认修复；指纹变化必须人工核对。
- 不要因登录失败就清空 ~/.ssh、删除 authorized_keys、重装 sshd 或批量杀进程。本文只指导诊断和安全切换。
- 服务启动、停止、健康检查和多机回滚统一调用 vllm-ascend-serving 脚本，避免手工 SSH 绕过转义、状态持久化和清理逻辑。

## 8. 交付前应记录的结果

一次成功的登录预检至少应产出：

| 项目 | 示例结果 |
| --- | --- |
| 节点标识 | hostname 返回值 |
| SSH 身份 | user@host:port |
| 连接模式 | docker / conda / direct |
| 执行上下文 | 容器名或 Conda 环境，direct 则为目标自身 |
| 权限 | root 或普通用户 + sudo 可用/不可用 |
| 平台 | npu-smi 可用性和设备摘要 |
| CANN/vLLM | set_env.sh 路径、which vllm、Python 导入结果 |
| 模型与网络 | 模型路径存在、服务/DP 端口策略 |
| 多机状态 | 每个 worker 的逐节点预检结果 |
| 下一步 | 可进入开箱检查，或明确阻塞项和切换建议 |

只有这些信息完整，才进入模型加载、服务启动、AISBench 压测或多机通信验证。

## 参考文件

- [feature-matrix.md](feature-matrix.md)
- [multi-node.md](multi-node.md)
- [failure-diagnosis.md](failure-diagnosis.md)
- [execution-notes.md](execution-notes.md)
