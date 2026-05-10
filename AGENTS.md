# <Project Name> — Agent Instructions

> 本文件会被 Claude Code 和 Codex 自动加载。任何 agent 开始工作前必须通读。

## 外部知识库

**顶层入口：** `~/.agent-knowledge/AGENTS.md`

**本项目加载的角色：**
- `default` <!-- 随着积累可以替换为更专门的角色 -->

Agent 必须先读顶层 AGENTS.md，再读本项目加载的每个角色的 `roles/<role>/AGENTS.md`。

## 本项目硬性规则

<!-- 只写"违反就可能造成不可逆损失"的规则，保持极度精简 -->

- 写操作在独立 worktree 进行
- Agent 不得自行 merge PR
- <按项目实际情况增加>
- 

## 本项目环境描述

<!-- 环境/配置类信息，因为会变，写在这里而不是知识库 -->

<!-- 示例：
- 部署环境：<...>
- 主要依赖版本：<...>
- 测试命令：`pnpm test`
- 启动命令：`pnpm dev`
-->
- vllm专属环境需要使用conda activate vllm启动
- vllm模型测试权重路径在/root/autodl-fs/AIinfra，其中有1.5B权重和8B权重

## 项目专属知识（可选）

<!-- 如果这个项目有大量专属知识不适合提炼到通用知识库，可以用 .agent-knowledge-project/ 本地目录存放，但仍然要遵循五分法的文件结构 -->
## 当前执行任务内容
- 把 vLLM 里的 Triton kernel 从 tl.load(ptr+offset, mask=...) 改写成 tl.make_block_ptr + tl.advance
- SwiGLU (~25 行) → ranks → RMSNorm → log-softmax → MRoPE

## 验证要求
- 每个 kernel 的数值等价测试（fp16/bf16/fp32，多种 shape）
- 5090 GPU上性能基准，确认无回归
- vllm.LLM.generate() 集成测试，验证 token 输出一致

## 默认规则

### 工作原则

- 非微小改动先说明方法。
- 需求有歧义、风险高或影响大时，先澄清并获批，再开始写代码。
- 坚持 Spec Coding，避免 Vibe Coding；Plan 只写方案、范围、风险和验收标准，不写实现代码。
- 优先小步迭代；实现与审查分离。
- 完成后可执行 /simplify；必要时使用 /loop。

### 编码约束

- 代码中只使用英文。
- 注释说明意图、约束和边界，不记录开发过程式说明。
- 优先用概念、模块、职责和符号名定位代码；不要只依赖易漂移的行号，必要时可补充文件路径。
- Spec 不依赖行号定位代码。
- 不为未被请求的未来需求提前抽象、泛化或暴露配置。

### 质量与验证

- 项目早期只保留最小必要质量标准：可运行、可验证、可回滚。
- 关键路径、高风险改动和外部接口必须可验证。
- 修复 bug 时，先复现，再修复，再验证。
- 任何"已完成""已修复""已通过"的结论，都必须附验证方式、命令或结果摘要。
- 若当前无法验证，必须明确说明原因、风险和未覆盖范围。

### 拆分与沉淀

- 将任务拆成低耦合、可独立验证的子任务；必要时使用 /batch。
- 重复出现且边界稳定的流程，应沉淀为 Skill、脚本或检查清单。
- 公共规则优先沉淀为文档、测试或自动化，而不是只停留在对话里。

### 协作与纠错

- 被纠正时，先验证问题是否适用于当前代码库，再调整做法。
- 外部建议先核对是否适用，再决定是否采纳。
- 对重复性问题，沉淀为明确规则、测试或自动检查。

### Codex 协作

- Codex 是补充能力，不是默认执行者；当前 Agent 负责主线推进、需求澄清、关键决策、首轮实现和最终验收。
- 仅在以下场景使用 Codex：独立只读代码评审、adversarial review、边界清晰且可并行的子任务、或长耗时调查与后台续跑；委派前必须先定义目标、约束、验收标准和边界。
- 不要把需求澄清、方案收敛、架构取舍、小而集中的直接实现或需要持续用户交互的主线任务交给 Codex；Codex 的结果必须由当前 Agent 整合并复核。
- 可用命令：`/codex:review`、`/codex:adversarial-review`、`/codex:rescue`、`/codex:status`、`/codex:result`、`/codex:cancel`

### 禁止事项

- 永远不要使用 /init，除非项目明确要求。
- CLAUDE.md 必须按项目实际需求编写，不套用空泛模板。
- 不要在代码注释、commit message 或 PR body 中使用描述开发进度的词，如 FIXED、Step、Week、Section、Phase、AC-x。
- 不要在代码注释、commit message 或 PR body 中出现 AI 工具名称，如 Codex、Claude、Grok、Gemini 等。
- 不要把外部实现细节、外部文档或外部技能树直接提升为当前项目的硬约束。