# Profile 字段说明

Profile 把平台相关的文字规则外部化为 JSON，按 App 适配、零代码扩展。复制本结构新增 `<app>.json`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | profile 名称 |
| `expected_package` | string | 期望前台包名（用于校验/审计） |
| `forbidden_markers` | string[] | **永不自动点击**的控件文字（黑名单，安全校验拦截） |
| `payment_stop_markers` | string[] | 出现即交人工的支付/收银页标志 |
| `manual_stop_markers` | string[] | 出现即交人工：验证码、人机验证、密码等 |
| `success_stop_markers` | string[] | 出现即交人工确认的成功页标志（如「支付成功」） |
| `loading_markers` | string[] | 加载/排队 → 等待 |
| `recovery_markers` | string[] | 错误/异常上下文 |
| `recovery_actions` | string[] | 错误上下文下优先点击的按钮（重试/确定…），按优先级排列 |

未提供的字段使用 `phonepilot/profile.py` 中的通用默认值。

`payment_stop_markers + manual_stop_markers + success_stop_markers` 合称「硬停止标志」：
主循环每帧扫描，命中立即保存截图并交人工接管。
