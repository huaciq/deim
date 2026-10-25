# SARDet 验证 DataLoader 的文件描述符限制修复

故障运行：
`/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop_seed0_20261025_135702_Hv6div`。
epoch 0 训练完成，last.pth 已保存；首次验证中一个 rank 在
`torch.multiprocessing.reductions.rebuild_storage_fd -> recvfds` 报
`RuntimeError: received 0 items of ancdata`。另一个 rank 随后在 COCO 结果汇总时等待，
最终产生分布式超时。这些超时是首个数据加载错误的后果。

服务器实测默认 RLIMIT_NOFILE=(1024,65535)，共享内存 252 GB、使用约 370 MB，
张量共享策略为默认 file_descriptor。验证每卡 batch=32、workers=4，默认每 worker
预取两批，每个 target 有多个独立 tensor，存在文件描述符不足的直接条件。
这次验证 batch 从先前协议的全局 32 恢复到官方的全局 64，增加了预取共享张量数量。

[PyTorch 2.1 官方说明](https://docs.pytorch.org/docs/2.1/multiprocessing.html#file-descriptor-file-descriptor)
指出此共享策略会持续占用较多描述符，并说明低上限场景需提高上限或改用另一共享策略。
本次选择提高软上限至现有硬上限 65535，保留原张量共享策略。

## 改动范围

- 在正式启动脚本 exec torchrun 前设置 `ulimit -Sn 65535`，由两个 rank 和各自 worker 继承。
- 在恢复、独立评估的命令中同样设置该上限。
- 不修改系统全局限制、模型、优化器、增强、训练/验证 batch、workers 或数据划分。
- 新运行采用相同固定 120 轮协议，从 COCO 初始化、使用新目录；故障日志和权重保留。

固定配置仍为
`configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32_paper_nozoom_nocrop.yml`。
此改动只涉及启动环境和文档，无训练引擎修改。

## 更新、重新启动、恢复、评估与监控

完整可复制命令见 [固定协议交接](sardet_paper120_nozoom_nocrop.md)，包含 Git 安全更新、
环境激活、COCO 初始化、冻结 resolved_config、nohup、保存 PID/disown、恢复及检查点评估。
原子集已准备完成，无需数据准备。

在服务器执行新的固定协议启动块前先更新：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
ulimit -Sn 65535
ulimit -Sn
```

GitHub HTTPS 无法连接时，经 Git bundle 的 fetch 和 --ff-only merge 同步提交，保留无关文件。
重连后明确设置新 OUT，具体运行目录、PID、实际验证结果写入本次交付 RUN.md。

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次新目录
cat "/proc/$(cat "$OUT/launcher.pid")/limits" | grep 'Max open files'
tail -f "$OUT/console.log"
```

```bash
OUT=/root/zxy/experiments/替换为本次新目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
tail -n 3 "$OUT/log.txt"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

## 针对此故障的功能验证

使用故障运行实际保存的 epoch-0 last.pth，GPU 0、1，保持全局验证 batch=64、
workers=4、同一 4,096 张验证图，设置软上限后完整执行独立 COCO bbox 评估。
这验证的是此前崩溃的完整验证路径。验证通过只说明该次路径正常，完整 120 轮训练
仍以新运行日志为准；不将未完成的训练描述为已成功。
