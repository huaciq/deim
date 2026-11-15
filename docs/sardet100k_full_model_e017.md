# E017：完整模型 = E013-v2 方向融合（主线）+ E016 RegionDenseSAR v3（辅助）

日期：2026-11-15。第一次把主线与辅助方法一起训练，检验两者是否叠加。

## 配置

`configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_region_dense_adaptive_36e_b32.yml` = E016 配置 + E013-v2 的
`DEIM.orientation_fusion`（projected、vflip、hidden 256、3 层、groups 8、reduction 4、init_bias 2.0、checkpoint_views）。
`tests/test_full_model_e017_config.py` 检查：相对 E016 只多出方向融合；方向融合各项与 E013-v2 相同；相对 E013-v2 只有训练数据构造不同
（RegionDenseSAR v3 计划代替 Mosaic/MixUp）。复用 E016 已生成并核对过的 v3 计划
`/root/zxy/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32`。

其余与 E000 36 轮协议一致：seed 0，全局 batch 32（GPU 4、5 各 16），COCO-M 初始化 `-t deim_dfine_hgnetv2_m_coco_90e.pth`，
AdamW 4e-4 / backbone 4e-5，flatcosine（warmup 2000、flat 20、no-aug 4），EMA 0.9999，AMP，多尺度 480–800 至 epoch 32，每轮完整 val。

## 判读

对照（同服务器、单种子）：E013-v2 val 67.26 / test 64.74，E016 val 66.33 / test 64.12，E000 val 65.29 / test 62.39。
- 组合 val 明显高于 E013-v2：互补成立，作为论文完整模型。
- 组合与 E013-v2 持平或更低：两者不叠加，需要先找原因（例如方向视角与区域拼图的相互作用），再定结构。
按类别重点看：tank、小目标（E016 的贡献）是否在组合中保留；bridge、harbor（E013-v2 的贡献）是否保留。
