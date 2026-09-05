# DarwinTrade

**输入股票代码，获得有分析依据的多空组合研究报告。**

[English](README.md) · [研究与复现文档](docs/RESEARCH_ZH.md) · [MIT 许可证](LICENSE)

DarwinTrade 将市场、新闻、社交和基本面分析，与确定性组合分配器、跨交易日记忆结合。网页可查看市场状态、建议权重、参考价格、分析理由和风险标记，并下载 JSON 报告。

输出是方向性信号和建议配置，不是保证兑现的未来股价。系统不连接券商，也不自动下单；置信度是模型分数，不代表经过校准的盈利概率。

## 快速启动

建议使用 Python 3.11 或 3.12。前端无需 Node.js 和构建步骤。

```bash
git clone https://github.com/Jacob6812/DarwinTrade.git
cd DarwinTrade
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

macOS / Linux：

```bash
source .venv/bin/activate
cp .env.example .env
```

安装：

```bash
python -m pip install -e .
```

在 `.env` 中填写自己的模型和行情服务配置：

```dotenv
LLM_BASE_URL=https://your-provider.example.com/v1
LLM_API_KEY=your-key
LLM_MODEL=your-provider-model-name
POLYGON_API_KEY=your-market-data-key
FINNHUB_API_KEY=your-finnhub-key
DARWINTRADE_DATA_MODE=auto
```

模型服务需兼容 OpenAI Chat Completions 和结构化响应。行情覆盖范围、历史数据、调用额度取决于你的服务商权限。仓库不包含密钥和行情缓存；缺失服务可能导致分析失败或部分证据不可用。

```bash
darwintrade serve
```

在浏览器打开 **http://127.0.0.1:8000**。找不到命令时使用 `python -m darwintrade.live.cli serve`。请在包含 `.env` 的目录中启动。

无需密钥也可点击 **Explore sample report** 浏览示例报告。示例价格和权重均为演示数据，不调用外部服务，也不改变会话。

## 使用流程

1. 输入美股代码，例如 `AAPL, MSFT, NVDA`，或点击示例组合。每次最多分析 30 只不同股票。
2. 输入初始资金（美元），日期留空使用最近的美国交易日，也可指定行情服务覆盖的历史交易日。
3. 点击 **Run analysis**。页面显示等待时间；实际耗时取决于股票数量、服务延迟和重试。分析期间保持页面打开。
4. 查看市场状态、建议多空权重、带正负号的名义金额、参考价、分析理由和风险标记。无法取得价格的股票会单独提示。没有持仓建议也是一种可能结果。
5. 点击 **Download JSON** 保存完整报告。

参考价优先采用指定交易日开盘价，缺失时可能使用之前的缓存收盘价，并非实时成交报价。市场趋势和收益历史使用之前的 K 线。历史日期分析不保证每个外部信息源均提供严格的历史时点快照。

### 跨日记忆

勾选 **Keep memory across runs** 后继续使用同一会话。浏览器刷新后会恢复会话编号，记忆文件保存在服务器。继续分析必须选择更晚的交易日；比较同一天、修改初始资金、分析更早日期时，点击 **New session**。

继续会话会按此前建议组合的价格变化累计模拟净值，资金输入框随之锁定。模拟净值不等于真实账户余额，不包含真实成交、手续费、借券成本和滑点。首次分析没有收益反馈，后续按一个交易日的释放延迟累计反馈。缺少价格的股票不会贡献收益，因此建议保持一致的股票列表。

## 命令行与 API

```bash
darwintrade predict AAPL MSFT NVDA --capital 100000 --date 2025-06-25
darwintrade predict AAPL MSFT NVDA --session SESSION_ID --date 2025-06-26
```

继续会话时省略 `--capital`，让净值延续。JSON 输出到 stdout，会话信息输出到 stderr。交互式 API 文档位于 **http://127.0.0.1:8000/docs**。

| 接口 | 功能 |
| --- | --- |
| `GET /api/health` | 配置状态和模型密钥是否存在 |
| `POST /api/decide` | 提交 `symbols`，可选 `trade_date`、`capital`、`session_id` |
| `POST /api/sessions` | 新建会话 |
| `GET /api/sessions` | 查看会话列表 |
| `GET /api/sessions/{id}` | 查看会话状态及近期历史 |

健康接口只检查配置，不代表服务商连接一定成功。请先用少量股票验证自己的服务权限。

## 数据与存储

默认行情缓存为 `storage/cache`，会话为 `storage/live/sessions`。可通过 `DARWINTRADE_CACHE_DIR`、`DARWINTRADE_SESSION_DIR` 或命令行 `--cache-dir`、`--session-dir` 修改。

`--offline` 或 `DARWINTRADE_DATA_MODE=offline_only` 只限制行情层读取缓存，**仍需要模型服务**，也不是无需配置的演示模式。全新克隆没有行情缓存。

同一会话目录请只运行一个服务进程；会话锁不跨进程共享。备份会话目录可保留学习记录。更多资金和风险参数见 [英文配置表](README.md#configuration-and-storage)。

## Docker

配置好 `.env` 后：

```bash
docker compose up --build
```

访问 **http://127.0.0.1:8000**。会话保存在命名卷中，行情缓存挂载到宿主机 `storage/cache`。服务以非 root 用户运行；Linux 下缓存目录需允许 UID 10001 写入。

当前版本适合本地使用。API 没有身份验证，分析会消耗服务额度；请保留默认的 localhost 绑定。如需提供网络访问，应由代理层提供认证和请求限制。

## 常见问题

| 情况 | 处理方式 |
| --- | --- |
| 提示未配置模型密钥 | 检查启动目录的 `.env`，填入配置后重启。 |
| 没有可用价格 | 检查代码、交易日、服务权限；离线模式需要已有缓存。 |
| 同日或更早日期被拒绝 | 新建会话，或选择更晚交易日。 |
| 分析失败 | 查看服务终端中的错误，检查模型名、接口地址、额度及网络。 |
| 分析等待较久 | 每个分析角色可能有多轮请求和重试；计时器不是完成进度百分比。 |
| 端口被占用 | 使用 `darwintrade serve --port 8001`。 |

## 开发与验证

```bash
python -m pip install -e ".[dev]"
python -m pytest -q --ignore=tests/test_offline_cache_completeness.py
python -m pip install build
python -m build --wheel
```

自动测试使用模型桩；依赖私有缓存的测试在缓存缺失时跳过。API 连续分析测试使用固定输入，实际执行会话、流水线与组合分配器。缓存完整性测试用于完整的研究数据集，不适用于刚克隆的仓库。离线测试通过不代表预测有效性或服务商可用性已经得到验证。

网页位于 `darwintrade/live/static/`，API 和会话位于 `darwintrade/live/`，研究回测引擎位于 `backtest/`。算法、消融及复现命令见 [研究文档](docs/RESEARCH_ZH.md)。

可选前端回归测试使用 Node.js 22：执行 `npm ci --ignore-scripts` 后运行 `npm test`。覆盖会话恢复、资金延续、示例隔离、重复提交、错误恢复和模型文本安全渲染。运行工具本身不需要 Node.js。

## 研究与引用

本项目实现 Jie Fang 与 Shaolei Zhang 的 *DarwinTrade: Hierarchical Self-Evolution for Multi-Agent Portfolio Trading*。引用信息见 [CITATION.cff](CITATION.cff)。采用 [MIT 许可证](LICENSE)。

本工具用于研究辅助，模型输出可能有误，不能替代独立判断。
