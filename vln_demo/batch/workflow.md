# 单个 DLC 的一键部署与评估

入口：`bash vln_demo/evaluate.sh`。一个完整 16 卡 DLC Pod 内完成模型、渲染、控制台、
实际验收及隔离批跑；不再创建 auto_cpu 任务，不依赖提交端 DSW 常驻。

```bash
bash vln_demo/evaluate.sh \
  --model next --datasets r2r,rxr --size ladder --workers 32 \
  --gpu-base /CPFS/任务记录/gpu-base.json \
  --output-dir /CPFS/任务记录/唯一运行名 --submit
```

不加 `--submit` 只生成 plan。`gpu-base.json` 为本人完整16卡任务的官方 detail 成功回执；
沿用同款 DSW 镜像、CPU/内存和挂载，普通优先级8。无需 `--cpu-base`。

| 参数 | 选择 |
|---|---|
| `--model` | `next`、`qwen36-35b`、`qwen38-27b`、`qwen35-122b`、`anonymous-data/<已注册模型ID>` |
| `--datasets` | `r2r`、`rxr`、`r2r,rxr`、`objectnav` |
| `--size` | 每数据集 `100`、`200`、`500`、`1000`、`full`；默认 `ladder` 逐档扩展 |
| `--workers` | 1–32，默认32个独立回合滚动并发 |
| `--manifest` | ObjectNav 必填，使用仓库版本化清单 |

每个模型独享一套服务。GPU0–3四路3DGS渲染，4–11待测模型TP8（网关模型则部署Next），
12–13底层35B，14 LightNav，15 Habitat-gs。主模型选择不改变底层35B/ReAct/think配置。
SGLang 参数仍只维护在 `vln_demo/vllm/sglang-*-tp8.sh`。
本地服务全部绑定同一个 job/Pod 的 `127.0.0.1`；网关模型沿用项目 provider 和鉴权。

每回合独立 workspace、HOME、临时目录和仿真状态，只共享无状态渲染及模型服务。
仓库只读、仅 workspace 可写，网络读取不受额外限制。总工具预算100，无回合墙钟限制，
单次移动600秒；中间帧省略渲染，最终观察帧照常返回。RxR 碰撞体及评分口径见 README。

## 生命周期与归档

supervisor 就绪后启动受管评估进程。文本、图像、工具、真实运动、模型路由和隔离验收通过，
才开始本机批跑。评估成功必须同时满足进程正常退出和 `_SUCCESS.json` 归档凭据；
失败会保留原因并退出，不让空闲服务永久占卡。supervisor 验证完成凭据后清理自己的所有
子进程并退出，由 DLC 回收整项资源。服务中途故障同样停止本次批跑；其他任务不受影响。

服务日志写数据盘；会话、图像、轨迹与指标仍写
`SpatialOnboarding/artifacts/<运行名>/`。plan 相邻的 `status.json`、`services.json`、
`evaluation.json`、`evaluation-ended.json` 记录绑定、验收和结果；服务 `status.json`
保存清理结果。进程结束、归档完成和平台资源释放分别核验，不把 HTTP 就绪当评估完成。

`--resume <plan.json> --submit` 只核对已有提交回执，避免重复创建；终止的任务不会因此
重建。重新评估使用新 plan；旧双任务 plan 已移除，不支持自动迁移。
共享源码、配置、提示词和 pi-agent 的版本/hash 变化仅记录，不中止或阻止扩档；启动版本保留，差异见 `version-drift.jsonl`。固定清单与归档仍校验完整性。此次 RxR 修复已从零重跑，不导入修复前的旧结果。
平台需要额外档位审批时保留真实申报草稿，不伪装训练、不自动发起审批。
