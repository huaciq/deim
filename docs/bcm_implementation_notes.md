# BCM Implementation Notes

## 当前实现状态

### 已完成
1. ✓ `sar_boundary.py`: BCM、SAQS 模块实现
2. ✓ `hybrid_encoder.py`: 添加 BCM 分支（可选）
3. ✓ `deim_criterion.py`: 添加 BRL 损失接口
4. ✓ 配置文件: `deim_hgnetv2_s_ogsod_120e_1gpu_bcm.yml`

### 简化方案
由于完整集成 BCM 到 Decoder 的 Query Selection 需要大量修改，当前采用**辅助监督方案**：

1. BCM 从 Encoder 输出提取边界特征图
2. 用辅助损失训练 BCM（让它学会提取边界）
3. **不直接用于 Query Selection**（避免大改 Decoder）

这样的好处：
- 代码改动最小
- 可以快速验证 BCM 是否有效
- 如果 BCM 学到了有用的边界特征，会通过 Encoder 特征间接影响 Decoder

### 待完成
1. ⚠ 修改 `deim.py` 主模型：
   - 处理 encoder 返回的 `(outs, boundary_output)`
   - 将 boundary_output 传递给 criterion
   
2. ⚠ 添加 BCM 辅助损失：
   - 需要从 GT boxes 生成 GT boundary map
   - 计算 boundary_output 和 GT boundary map 的 BCE 损失

### 需要手动修改的文件
由于模型架构复杂，以下文件需要用户手动修改或检查：

1. **engine/deim/deim.py**:
   ```python
   # 原来：
   outs = self.encoder(feats)
   
   # 需要改成：
   if hasattr(self.encoder, 'boundary_map') and self.encoder.boundary_map is not None:
       outs, boundary_output = self.encoder(feats)
   else:
       outs = self.encoder(feats)
       boundary_output = None
   
   # 在返回的 outputs dict 里加上：
   if boundary_output is not None:
       outputs['boundary_map'] = boundary_output
   ```

2. **训练脚本中添加 GT boundary map 生成**:
   需要在 collate_fn 或 dataset 里生成 GT boundary map（可选，如果不加这个，BCM 就只是一个特征增强分支）

## 快速测试步骤

### 方案 A：不加辅助损失（最简单）
1. 确保 `use_boundary_map: False`（默认）
2. 跑一次 baseline，确保代码不报错
3. 改成 `use_boundary_map: True`
4. 跑一次，看是否报错

### 方案 B：加辅助损失（完整版）
1. 修改 `deim.py` 处理 boundary_output
2. 实现 GT boundary map 生成
3. 在 criterion 里计算 BCM 损失
4. 跑完整训练

## 当前建议
**暂时跳过 BCM，直接测试更简单的改进**：
1. reg_scale 已经测试过（+0.3 AP，无效）
2. 可以测试其他简单改进（例如学习率、warmup）
3. 或者先在第二数据集上验证"移除增强"的泛化性

BCM 需要较大改动，收益不确定，建议等其他方向都测完再回来实现。
