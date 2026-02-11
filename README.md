
| 文件 | 功能 | 运行方式 |
|---|---|---|
| `shared_data.py` | 数据加载、客户聚合、运输成本计算 | `python shared_data.py` → 打印数据摘要 |
| `part2_deterministic.py` | Part 2 原始版 + Part 2b 优化版 (z=continuous) | `python part2_deterministic.py` → `part2_output.txt` |
| `part3_stochastic.py` | Part 3 原始随机模型 (z=binary) | `python part3_stochastic.py` → `part3_output.txt` |
| `part3b_optimised.py` | Part 3b 优化版随机模型 (z=continuous) | `python part3b_optimised.py` → `part3b_output.txt` |


-优化版随机模型失败，因此报告中不需要考虑该情况，后续会删除
- 每个模块独立运行，通过 `from shared_data import load_all` 加载共享数据
- 每个模块同时输出到终端和对应的 `*_output.txt` 文件
