# Bug 分析：alpha_reactor1-9 拟合结果全为0

## 现象

运行 `fitter/minuit.py` 进行 Minuit 拟合后，9个反应堆不相关不确定性参数 `alpha_reactor1` 到 `alpha_reactor9` 的拟合结果全部为0，没有被有效拟合。

## 根本原因：参数传递顺序错误

在 `fitter/src/fitter/analysis/fitter.py` 的 `_chi2_standard` 方法中，调用 `self.rea.get_E_p_spectrum()` 时使用了**位置参数**传递，但传入顺序与函数签名定义的顺序不一致。

### 调用顺序（fitter.py 第456-477行，修复前）

```python
T_nu = self.rea.get_E_p_spectrum(
    dmsq31,           # → dmsq31        ✓
    sinsq12,          # → sinsq12       ✓
    dmsq21,           # → dmsq21        ✓
    sinsq13,          # → sinsq13       ✓
    alpha_Eres_a,     # → alpha_l_pull0  ✗ 错误！
    alpha_Eres_b,     # → alpha_l_pull1  ✗ 错误！
    alpha_Eres_c,     # → alpha_l_pull2  ✗ 错误！
    alpha_l_pull0,    # → alpha_l_pull3  ✗ 错误！
    alpha_l_pull1,    # → alpha_Eres_a   ✗ 错误！
    alpha_l_pull2,    # → alpha_Eres_b   ✗ 错误！
    alpha_l_pull3,    # → alpha_Eres_c   ✗ 错误！
    alpha_reactor1,   # → alpha_rho_ME   ✗ 错误！
    alpha_reactor2,   # → alpha_reactor1 ✗ 错位！
    alpha_reactor3,   # → alpha_reactor2 ✗ 错位！
    ...
    alpha_reactor9,   # → alpha_reactor8 ✗ 错位！
    # alpha_reactor9 从未收到任何值，始终为默认值0
)
```

### 函数签名（reactor_expected.py 第439-461行）

```python
def get_E_p_spectrum(
    self,
    dmsq31, sinsq12, dmsq21, sinsq13,
    alpha_l_pull0, alpha_l_pull1, alpha_l_pull2, alpha_l_pull3,  # 位置5-8
    alpha_Eres_a, alpha_Eres_b, alpha_Eres_c,                   # 位置9-11
    alpha_rho_ME,                                                 # 位置12
    alpha_reactor1, ..., alpha_reactor9,                          # 位置13-21
):
```

### 错位导致的后果

| Minuit 调整的参数 | 实际影响的函数参数 | 后果 |
|---|---|---|
| alpha_Eres_a/b/c | alpha_l_pull0/1/2 | 能量分辨率参数实际在调非线性 |
| alpha_l_pull0..3 | alpha_l_pull3, alpha_Eres_a/b/c | 非线性参数实际在调能量分辨率 |
| alpha_reactor1 | alpha_rho_ME | 反应堆参数1实际在调物质效应 |
| alpha_reactor2..9 | alpha_reactor1..8 | 依次错位一个位置 |
| （无） | alpha_reactor9 | 该参数始终为0 |

由于参数错位：
1. Minuit 调整 `alpha_reactor` 参数时，实际改变的是其他物理参数
2. chi2 对 `alpha_reactor` 的有效梯度近乎为零
3. 约束项 `alpha_reactor_i^2 / sigma^2` 惩罚非零值
4. 因此所有 `alpha_reactor` 参数收敛到 0

## 修复方案

将位置参数调用改为**关键字参数**调用：

```python
T_nu = self.rea.get_E_p_spectrum(
    dmsq31=dmsq31,
    sinsq12=sinsq12,
    dmsq21=dmsq21,
    sinsq13=sinsq13,
    alpha_Eres_a=alpha_Eres_a,
    alpha_Eres_b=alpha_Eres_b,
    alpha_Eres_c=alpha_Eres_c,
    alpha_l_pull0=alpha_l_pull0,
    alpha_l_pull1=alpha_l_pull1,
    alpha_l_pull2=alpha_l_pull2,
    alpha_l_pull3=alpha_l_pull3,
    alpha_reactor1=alpha_reactor1,
    ...
    alpha_reactor9=alpha_reactor9,
)
```

使用关键字参数后，参数传递不依赖位置顺序，避免了此类错误。

## 教训

当函数参数数量较多（>5个）时，应**始终使用关键字参数**进行调用，避免因位置顺序不匹配导致难以发现的 bug。
