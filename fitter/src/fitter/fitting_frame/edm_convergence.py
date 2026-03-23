"""
EDM (Estimated Distance to Minimum) 收敛判断条件实现
基于iMinuit中的EDM计算方法，应用于SGD优化器
"""

import torch
import numpy as np
from typing import Tuple, Optional
from loguru import logger

torch.set_default_dtype(torch.float64)


class DiagEDMTracker:
    """
    对角Hessian近似的EDM跟踪器

    使用类似Adam优化器的二阶矩估计来近似Hessian的对角元素
    """

    def __init__(self, param_shape, beta2=0.999, eps=1e-8):
        """
        Parameters:
        -----------
        param_shape : tuple or int
            参数形状
        beta2 : float
            二阶矩估计的指数衰减率，默认0.999
        eps : float
            数值稳定性的小常数，默认1e-8
        """
        if isinstance(param_shape, int):
            param_shape = (param_shape,)
        self.v = torch.zeros(param_shape, dtype=torch.float64)  # 二阶矩估计
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def update(self, grad):
        """
        更新二阶矩估计

        Parameters:
        -----------
        grad : torch.Tensor
            当前梯度

        Returns:
        --------
        torch.Tensor : 偏差校正后的二阶矩估计
        """
        # 确保梯度与v的形状匹配
        if grad.shape != self.v.shape:
            # 如果形状不匹配，重新初始化
            self.v = torch.zeros_like(grad, dtype=torch.float64)

        self.t += 1
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # 偏差校正
        v_hat = self.v / (1 - self.beta2**self.t)
        return v_hat

    def compute_edm(self, grad, v_hat):
        """
        计算对角近似EDM

        EDM_diag ≈ 0.5 * sum( g_i^2 / (v_hat_i + eps) )

        Parameters:
        -----------
        grad : torch.Tensor
            当前梯度
        v_hat : torch.Tensor
            偏差校正后的二阶矩估计

        Returns:
        --------
        float : EDM值
        """
        return 0.5 * torch.sum(grad**2 / (v_hat + self.eps)).item()


class EDMConvergence:
    """
    EDM (Estimated Distance to Minimum) 收敛判断类

    EDM 是 iMinuit 中用于判断拟合收敛的核心指标。

    数学定义：
    EDM = 1/2 * g^T * H^(-1) * g

    其中：
    - g: 梯度向量 (gradient)
    - H: Hessian矩阵（二阶导数矩阵）
    - H^(-1): Hessian的逆矩阵

    物理意义：
    EDM 估计了当前参数位置到真实最小值的"距离"。
    它考虑了函数的曲率信息（通过Hessian矩阵），给出了一个标量值来表征收敛程度。

    在实际应用中（使用iMinuit标准 EDM < tolerance * 2 * UP）：
    - 对于χ²拟合（UP=1.0）：
      * EDM < 2e-4: 标准收敛（tolerance=1e-4）
      * EDM < 2e-5: 高精度收敛
      * EDM < 2e-6: 极高精度收敛
    - 对于负对数似然拟合（UP=0.5）：
      * EDM < 1e-4: 标准收敛（tolerance=1e-4）
      * EDM < 1e-5: 高精度收敛
      * EDM < 1e-6: 极高精度收敛
    """

    def __init__(self, tolerance: float = 1e-4, up: float = 1.0):
        """
        初始化EDM收敛判断器

        Parameters:
        -----------
        tolerance : float
            EDM收敛阈值，默认1e-4（iMinuit默认值）
        up : float
            误差定义（UP value），对于最小二乘拟合up=1.0，
            对于负对数似然拟合up=0.5
        """
        self.tolerance = tolerance
        self.up = up
        self.history = []
        self.diag_tracker = None  # 对角Hessian跟踪器

    def compute_edm(
        self, gradient: torch.Tensor, hessian: Optional[torch.Tensor] = None, inv_hessian: Optional[torch.Tensor] = None
    ) -> float:
        """
        计算EDM值

        Parameters:
        -----------
        gradient : torch.Tensor
            目标函数的梯度向量
        hessian : torch.Tensor, optional
            Hessian矩阵，如果未提供inv_hessian则需要
        inv_hessian : torch.Tensor, optional
            Hessian的逆矩阵，如果已经计算好可以直接提供

        Returns:
        --------
        float : EDM值
        """
        g = gradient.flatten()

        # 如果没有提供逆Hessian，则计算它
        if inv_hessian is None:
            if hessian is None:
                raise ValueError("必须提供hessian或inv_hessian之一")

            # 确保Hessian是对称的
            H = 0.5 * (hessian + hessian.T)

            # 添加小的正则化项以确保数值稳定性
            eps = 1e-8
            H_reg = H + eps * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

            try:
                # 使用Cholesky分解求逆（对于正定矩阵更稳定）
                L = torch.linalg.cholesky(H_reg)
                inv_H = torch.cholesky_inverse(L)
            except:
                # 如果Cholesky失败，使用标准求逆
                inv_H = torch.linalg.inv(H_reg)
        else:
            inv_H = inv_hessian

        # 计算 EDM = 0.5 * g^T * H^(-1) * g
        edm = 0.5 * torch.dot(g, torch.matmul(inv_H, g)).item()

        # 注意：EDM的标准定义不包含UP值
        # UP值在收敛判断时使用：EDM < tolerance * 2 * UP

        return edm

    def compute_edm_approximate(self, gradient: torch.Tensor, parameters: torch.Tensor, learning_rate: float) -> float:
        """
        使用修正的近似方法计算EDM（基于对角Hessian近似）

        修正说明：
        1. 移除了错误的 learning_rate * ||g||² 公式
        2. 使用对角Hessian近似，基于Adam风格的二阶矩估计
        3. 提供回退方案当跟踪器未初始化时

        Parameters:
        -----------
        gradient : torch.Tensor
            梯度向量
        parameters : torch.Tensor
            当前参数值
        learning_rate : float
            当前学习率（用于回退方案）

        Returns:
        --------
        float : 近似的EDM值
        """
        g = gradient.flatten()

        # 初始化对角跟踪器（如果还没有）
        if self.diag_tracker is None:
            self.diag_tracker = DiagEDMTracker(g.shape)

        # 使用对角Hessian近似方法（推荐）
        try:
            v_hat = self.diag_tracker.update(g)
            edm = self.diag_tracker.compute_edm(g, v_hat)

            # 数值稳定性检查
            if torch.isnan(torch.tensor(edm)) or torch.isinf(torch.tensor(edm)) or edm <= 0:
                raise ValueError("EDM calculation resulted in invalid value")

        except Exception as e:
            # 回退方案：使用梯度范数的保守估计
            logger.warning(f"对角Hessian近似失败，使用回退方案: {e}")

            # 回退方法1：简单的梯度范数估计
            edm_fallback1 = 0.5 * torch.norm(g).item()

            # 回退方法2：考虑参数规模的相对估计
            param_norm = torch.norm(parameters.flatten())
            if param_norm > 1e-8:
                relative_grad_norm = torch.norm(g).item() / param_norm.item()
                edm_fallback2 = 0.5 * relative_grad_norm * param_norm.item() * 0.1
            else:
                edm_fallback2 = edm_fallback1

            # 使用更保守的估计
            edm = min(edm_fallback1, edm_fallback2)

            # 对于大学习率，应用额外的保守因子
            if learning_rate > 1.0:
                edm *= min(1.0, 1.0 / learning_rate)

        # 确保EDM值在合理范围内
        edm = max(edm, 1e-12)  # 避免过小的值
        edm = min(edm, 1e6)  # 避免过大的值

        return edm

    def compute_edm_from_loss_history(self, loss_history: list, param_history: list = None) -> float:
        """
        从损失函数历史估计EDM

        使用损失函数的变化率来估计收敛程度

        Parameters:
        -----------
        loss_history : list
            最近的损失函数值历史
        param_history : list, optional
            最近的参数值历史

        Returns:
        --------
        float : 估计的EDM值
        """
        if len(loss_history) < 3:
            return float("inf")

        # 计算损失函数的一阶和二阶差分
        delta_loss = loss_history[-1] - loss_history[-2]
        delta2_loss = (loss_history[-1] - loss_history[-2]) - (loss_history[-2] - loss_history[-3])

        # 避免除零
        if abs(delta2_loss) < 1e-10:
            return abs(delta_loss)

        # 使用泰勒展开估计EDM
        # EDM ≈ |Δf|^2 / (2 * |Δ²f|)
        edm_estimate = abs(delta_loss) ** 2 / (2 * abs(delta2_loss))

        # 如果有参数历史，使用参数变化进行修正
        if param_history and len(param_history) >= 2:
            param_change = torch.norm(param_history[-1] - param_history[-2])
            # 根据参数变化调整EDM估计
            edm_estimate *= 1 + param_change.item()

        # 注意：EDM的标准定义不包含UP值
        # UP值在收敛判断时使用：EDM < tolerance * 2 * UP
        return edm_estimate

    def check_convergence(self, edm: float) -> Tuple[bool, str]:
        """
        检查是否满足收敛条件

        使用iMinuit标准的收敛判据：EDM < tolerance * 2 * UP

        Parameters
        ----------
        edm : float
            当前的EDM值

        Returns
        -------
        Tuple[bool, str] : (是否收敛, 收敛状态描述)
        """
        self.history.append(edm)

        # iMinuit标准收敛阈值：tolerance * 2 * UP
        threshold = self.tolerance * 2 * self.up
        threshold_high = threshold * 0.1  # 高精度阈值
        threshold_ultra = threshold * 0.001  # 极高精度阈值

        if edm < threshold_ultra:
            return True, f"极高精度收敛 (EDM={edm:.2e} << {threshold:.2e})"
        elif edm < threshold_high:
            return True, f"高精度收敛 (EDM={edm:.2e} < {threshold:.2e})"
        elif edm < threshold:
            return True, f"标准收敛 (EDM={edm:.2e} < {threshold:.2e})"
        else:
            return False, f"未收敛 (EDM={edm:.2e} > {threshold:.2e})"


def compute_edm_gradient_norm(grad_flat: torch.Tensor) -> float:
    """EDM approximation using gradient norm: EDM ≈ 0.5 * ||g||^2"""
    return 0.5 * grad_flat.dot(grad_flat).item()


def compute_edm_from_lbfgs_state(grad_flat: torch.Tensor, optimizer) -> float:
    """
    Compute EDM = 0.5 * g^T * H^{-1} * g using L-BFGS internal state.

    Extracts the stored curvature pairs from PyTorch's L-BFGS optimizer
    and applies the two-loop recursion to compute H^{-1} @ g without
    materializing the full inverse Hessian.

    Falls back to gradient-norm approximation if state is unavailable.
    """
    try:
        state = optimizer.state[optimizer._params[0]]
        old_dirs = state.get("old_dirs")  # y vectors (gradient diffs)
        old_stps = state.get("old_stps")  # s vectors (parameter diffs)
        ro = state.get("ro")              # 1 / (y · s)
        H_diag = state.get("H_diag")     # scalar initial Hessian scale
    except (AttributeError, KeyError, IndexError):
        return compute_edm_gradient_norm(grad_flat)

    if not old_dirs or len(old_dirs) == 0:
        return compute_edm_gradient_norm(grad_flat)

    # L-BFGS two-loop recursion to compute H_inv @ g
    k = len(old_dirs)
    q = grad_flat.clone()
    al = [0.0] * k

    for i in range(k - 1, -1, -1):
        al[i] = ro[i] * old_stps[i].dot(q)
        q.add_(old_dirs[i], alpha=-al[i])

    r = torch.mul(q, H_diag)

    for i in range(k):
        be = ro[i] * old_dirs[i].dot(r)
        r.add_(old_stps[i], alpha=al[i] - be)

    edm = 0.5 * grad_flat.dot(r).item()
    return max(edm, 0.0)  # EDM should be non-negative


def integrate_edm_with_sgd(loss_fn, optimizer, max_iterations=1000, edm_tolerance=1e-4, check_interval=10):
    """
    将EDM收敛判断集成到SGD优化过程中

    Parameters:
    -----------
    loss_fn : callable
        损失函数对象（需要支持backward）
    optimizer : torch.optim.Optimizer
        PyTorch优化器
    max_iterations : int
        最大迭代次数
    edm_tolerance : float
        EDM收敛阈值
    check_interval : int
        检查EDM的间隔（每多少次迭代检查一次）

    Returns:
    --------
    dict : 包含优化历史和收敛信息的字典
    """
    edm_checker = EDMConvergence(tolerance=edm_tolerance)

    loss_history = []
    edm_history = []
    gradient_norms = []
    converged = False
    converged_at = None

    for i in range(max_iterations):
        # 前向传播和反向传播
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()

        # 记录损失
        loss_value = loss.item()
        loss_history.append(loss_value)

        # 收集梯度
        gradients = []
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    gradients.append(p.grad.data.flatten())

        if gradients:
            gradient_vector = torch.cat(gradients)
            grad_norm = torch.norm(gradient_vector).item()
            gradient_norms.append(grad_norm)

            # 每隔一定间隔检查EDM
            if i % check_interval == 0 and i > 0:
                # 使用近似方法计算EDM
                current_lr = optimizer.param_groups[0]["lr"]

                # 收集所有参数
                params = []
                for param_group in optimizer.param_groups:
                    for p in param_group["params"]:
                        if p.grad is not None:
                            params.append(p.data.flatten())
                param_vector = torch.cat(params) if params else torch.tensor([])

                # 计算近似EDM
                edm_approx = edm_checker.compute_edm_approximate(gradient_vector, param_vector, current_lr)

                # 也可以使用损失历史估计
                if len(loss_history) >= 3:
                    edm_from_loss = edm_checker.compute_edm_from_loss_history(loss_history)
                    # 取两种估计的平均
                    edm = (edm_approx + edm_from_loss) / 2
                else:
                    edm = edm_approx

                edm_history.append(edm)

                # 检查收敛
                is_converged, status = edm_checker.check_convergence(edm)

                if i % (check_interval * 10) == 0:  # 每100次迭代打印一次
                    logger.info(
                        f"Iteration {i}: Loss={loss_value:.6f}, GradNorm={grad_norm:.6e}, EDM={edm:.6e}, {status}"
                    )

                if is_converged and not converged:
                    converged = True
                    converged_at = i
                    logger.info(f"✓ 收敛于第 {i} 次迭代: {status}")
                    break

        # 更新参数
        optimizer.step()

    return {
        "converged": converged,
        "converged_at": converged_at,
        "final_loss": loss_history[-1],
        "final_edm": edm_history[-1] if edm_history else None,
        "loss_history": loss_history,
        "edm_history": edm_history,
        "gradient_norms": gradient_norms,
        "iterations": len(loss_history),
    }


# iMinuit中EDM的核心计算逻辑（简化版）
class MinuitEDM:
    """
    iMinuit中EDM计算的核心逻辑（简化的Python实现）

    在iMinuit/Minuit2的C++源代码中，EDM的计算主要在：
    - VariableMetricBuilder::Estimate()
    - MnUserParameterState::Edm()

    关键步骤：
    1. 计算梯度 g = ∇f(x)
    2. 获取误差矩阵（协方差矩阵的逆）V = H^(-1)
    3. 计算 EDM = g^T * V * g / 2
    4. 收敛判断：EDM < tolerance * 2 * UP
    """

    @staticmethod
    def calculate_edm_minuit_style(gradient, error_matrix, scale_factor=1.0):
        """
        按照iMinuit的方式计算EDM

        Parameters:
        -----------
        gradient : numpy.ndarray
            梯度向量
        error_matrix : numpy.ndarray
            误差矩阵（Hessian的逆矩阵）
        scale_factor : float
            缩放因子（对应于UP值）

        Returns:
        --------
        float : EDM值
        """
        # 确保是numpy数组
        if torch.is_tensor(gradient):
            gradient = gradient.detach().cpu().numpy()
        if torch.is_tensor(error_matrix):
            error_matrix = error_matrix.detach().cpu().numpy()

        # iMinuit的EDM计算
        # EDM = 0.5 * g^T * V * g，其中V是误差矩阵
        edm = 0.5 * np.dot(gradient, np.dot(error_matrix, gradient))

        # 注意：标准EDM定义不包含UP值
        # scale_factor(UP值)在收敛判断时使用，不在EDM计算中使用
        # 这里保留scale_factor参数是为了向后兼容
        if scale_factor != 1.0:
            import warnings

            warnings.warn("建议不要在EDM计算中使用scale_factor，UP值应在收敛判断时使用", DeprecationWarning)
            edm *= scale_factor

        return edm

    @staticmethod
    def convergence_criteria_minuit(edm, tolerance=0.001, up=1.0):
        """
        iMinuit的收敛判断标准

        收敛条件：EDM < tolerance * 2 * UP * 0.001

        其中：
        - tolerance: 用户设定的容差（默认0.001）
        - UP: 误差定义（χ²拟合为1.0，负对数似然为0.5）
        - 0.001: 额外的安全因子
        """
        threshold = tolerance * 2 * up * 0.001
        return edm < threshold, threshold


# 示例：如何在您的SGD代码中使用EDM
def example_usage():
    """
    演示如何在SGD优化中使用EDM收敛判断
    """
    import torch.nn as nn

    # 创建一个简单的优化问题
    torch.manual_seed(42)

    # 目标：最小化 f(x,y) = (x-1)² + (y-2)²
    params = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))

    def loss_fn():
        x, y = params
        return (x - 1) ** 2 + (y - 2) ** 2

    # 设置优化器
    optimizer = torch.optim.SGD([params], lr=0.1)

    # 创建EDM检查器
    edm_checker = EDMConvergence(tolerance=1e-4)

    # 优化循环
    for i in range(100):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()

        # 计算EDM
        if i % 5 == 0:  # 每5次迭代检查一次
            grad = params.grad.clone()
            lr = optimizer.param_groups[0]["lr"]
            edm = edm_checker.compute_edm_approximate(grad, params, lr)

            converged, status = edm_checker.check_convergence(edm)
            print(f"Iter {i:3d}: Loss={loss.item():.6f}, EDM={edm:.6e}, {status}")

            if converged:
                print(f"✓ 优化收敛！最终参数: x={params[0].item():.6f}, y={params[1].item():.6f}")
                break

        optimizer.step()


if __name__ == "__main__":
    # 运行示例
    example_usage()
