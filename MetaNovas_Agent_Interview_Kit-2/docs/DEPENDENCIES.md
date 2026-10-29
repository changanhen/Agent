# 依赖与运行说明

## 运行依赖

| 依赖 | 用途 | 安装方式 |
|---|---|---|
| Python 3.8+，建议 3.10+ | 执行 Agent 和练习 runner | 安装 Python；当前本机为 3.8.6 |
| `interview_sdk.py` | 提供 `chat()` 与 `ModelError`，访问本地模型接口或正式代理 | 套件已提供，本地与 `agent.py` 放在同一目录；正式平台提供 |
| `json` | 序列化、工具参数与审核结果解析 | Python 标准库 |
| `hashlib` | 为精确写动作生成比较指纹 | Python 标准库 |
| `math`、`re` | 常见 JSON Schema 数值、字符串约束校验 | Python 标准库 |
| `decimal`、`datetime` | 精确算术、金额舍入及日期目标比较 | Python 标准库 |
| `copy.deepcopy` | 防止内部处理修改调用方传入的对话、工具定义 | Python 标准库 |
| `__future__.annotations` | 延迟求值类型注解，兼容 Python 3.8 | Python 标准库 |

没有第三方 pip 依赖。`requirements.txt` 仅记录说明，不需要安装
`openai`、`anthropic`、`deepseek`、LangChain 或数据库客户端。
SDK 自身使用的 `os`、`socket`、`urllib`、`typing` 也全部属于标准库。

## 本地配置与命令

本地实际模型练习需要自行设置密钥，例如 PowerShell：

```powershell
$env:INTERVIEW_PROVIDER = 'openai'  # 本地支持 openai 或 deepseek
$env:INTERVIEW_API_KEY = 'your-key'
# 可选：设置 INTERVIEW_MODEL，省略则使用 SDK 默认模型。
python -m scripts.practice_runner
```

不要把实际密钥写进源码或提交文件。本地练习需要网络访问和可用的服务账号。
正式运行的模型访问由平台代理提供，无需自行创建连接或访问外部服务。
只需提交 `agent.py`；其他新增文件用于本地说明和验证。

离线测试无需 API 密钥、网络或额外测试框架：

```powershell
python -B -m unittest discover -s tests -t . -v
```

## 函数与调用流程

当前项目目录、全部命令和测试报告说明以根目录的 [README.md](../README.md) 为准。
练习数据位于 `data/practice/`，本地脚本位于 `scripts/`，测试位于 `tests/`，报告位于 `reports/`。

`respond()` 每次从完整对话重新推导状态，不依赖跨调用进程内存。

新增 `reconstruct_intent()`：在生成动作之前提取逐项目标、仍有效的约束、
需求最后修改位置、目标状态以及精确写入授权，并校验用户消息来源。
`check_current_intent()` 会独立于审核结果拦截已撤销目标、旧提案授权、
跨写工具已消耗的确认，以及工具副作用分类不一致的情况。

1. `build_tool_catalog()`：索引运行时工具与参数 schema。
2. `replay_history()`：关联调用和结果，保留错误及没有结果的尝试；收到返回值不等于业务成功。
3. `build_decision_context()`、`build_model_messages()`：组织历史证据索引，保留完整原始对话。
4. `generate_candidate()`：通过 SDK 生成下一步。
5. `validate_response_shape()`、`validate_tool_arguments()`：校验返回契约及常见 schema 约束。
6. `review_candidate()`：通过一次模型调用审核意图、政策、证据、授权、批量进度及完成声明。
7. `check_action_preconditions()`、`check_confirmation()`、`check_batch_progress()`、
   `check_completion_claims()`：检查审核报告；写动作另检查提案/确认的消息位置和重复尝试。
8. `build_write_spec()`、`fingerprint_write()`：将精确参数和审核提取的费用等条款规范化；指纹不是授权凭证。
9. `repair_candidate()`：失败时最多修复一次；修复结果再次校验和审核。
10. `finalize_response()`：返回唯一回复或工具调用，由外部平台执行。

正常路径每轮三次模型请求（意图重建、动作生成、审核）；意图格式修复最多一次，动作修复最多一次，总计最多六次。每轮最多返回一个业务工具调用。
审核解析失败、未通过审核或发生 SDK 异常时，停止发出工具动作并返回简短说明。

最新的单文件实现还包含：最近用户指令与目标的关联检查、完成/受阻目标的证据验证、
首次批量写入前的逐记录查询和计划验证、最终答复的未完成目标检查，以及简写/旧式
工具消息的归一化。审核和意图状态都支持完整的 Markdown JSON 代码块。
模型的语义判断仍需真实 API 测试；离线测试不会替代正式评分。

## 实现边界

- 自然语言确认、工具副作用、政策适用性和跨工具授权消耗依赖模型审核，基础代码不提供形式化安全保证。
- 代码额外阻止：同一确认后再次发出完全相同的写调用；此前失败或没有结果也算已尝试。
- 本地 schema 校验覆盖类型、必填项、额外字段、嵌套对象/数组、枚举、常量、长度、正则、
  数值边界、`anyOf`/`oneOf`/`allOf`/`not` 和本地 `$ref`。不是完整 JSON Schema 验证器，
  其他约束仍需模型依据原始 schema 检查；不下载外部引用。
- 保留 SDK 返回的本地 DeepSeek `_reasoning_content` 传输字段，以便现有 runner 回放。
  根据套件 SDK 约定，正式代理不会向候选代码返回该字段。
- 离线测试通过 mock 模型响应验证控制流程和边界，不代表真实模型理解政策或正式题目得分已验证。
