# Project Context: Reactor Neutrino Fitter (Sub-project Recovery)

## 1. Project Overview
当前项目是从完整版 `reactor-neutrino-fitter` 中提取出的精简子项目。目标是在保持轻量化的同时，恢复必要的拟合功能（特别是 SGD 和 L-BFGS），并确保测试脚本能够正常运行。

## 2. Technical Context
- **Language:** Python
- **Core Task:** Neutrino oscillation parameter fitting.
- **Key Modules:** Tensorized fitting framework, EDM (Estimated Distance to Minimum) convergence testing.
- **Current Issue:** `test_edm_convergence.py` 运行失败，原因是底层代码缺少某些核心类或函数。

## 3. Reference Source (CRITICAL)
当你发现当前文件夹中的代码缺失功能、导入报错或逻辑不完整时，**必须参考以下路径中的原始代码**：
- **Reference Path:** `/data/juno/xuejq/paper/reactor-neutrino-fitter/fitter`
- **Reference Status:** 该路径下的代码是完整的，且其 `test_edm_convergence.py` 可以正常运行。

## 4. Development Tasks
1. **Implement Optimizers:** 在当前项目中增加/完善 `SGD` 和 `L-BFGS` 拟合方法。
2. **Fix Test Script:** - 运行 `python test_edm_convergence.py`。
   - 根据报错信息（缺少模块、参数不匹配等），去 `/data/juno/xuejq/paper/reactor-neutrino-fitter/fitter` 寻找对应的实现。
   - 将缺失的功能有选择性地迁移/重构到当前项目中，直到测试通过。
3. **Minimize Bloat:** 迁移代码时只引入必要的依赖，避免将整个原项目全部搬过来。

## 5. Guidelines
- **Import Handling:** 注意修改迁移代码后的 import 路径，确保符合当前子项目的目录结构。
- **Verification:** 每次修复一个报错后，重新运行测试脚本确认进度。
- **Optimizer Logic:** 确保 SGD 和 L-BFGS 的实现与现有的张量化框架兼容。