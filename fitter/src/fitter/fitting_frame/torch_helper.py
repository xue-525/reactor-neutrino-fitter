import time
import os
import json
import numpy as np
import torch
from loguru import logger
from typing import Optional, Dict, Any, Sequence, List, Tuple
from ..fitting_frame.edm_convergence import (
    EDMConvergence,
    compute_edm_from_lbfgs_state,
    compute_edm_gradient_norm,
)
from ..fitting_frame.model import ReactorLoss

torch.set_default_dtype(torch.float64)


def compute_hessian(loss_fn, cache_key=None, hessian_cache=None):
    """
    计算损失函数的Hessian矩阵

    Parameters
    ----------
    loss_fn : callable
        损失函数对象，需要支持parameters()方法
    cache_key : str, optional
        缓存键，用于避免重复计算
    hessian_cache : dict, optional
        Hessian缓存字典

    Returns
    -------
    torch.Tensor
        Hessian矩阵，形状为(n_params, n_params)

    Notes
    -----
    使用PyTorch的自动微分功能计算二阶导数矩阵。
    对于大型模型，此操作可能计算成本较高。
    1. 添加缓存机制，避免重复计算
    2. 使用更高效的计算方法
    3. 添加数值稳定性检查
    """
    # 检查缓存
    if cache_key is not None and hessian_cache is not None and cache_key in hessian_cache:
        return hessian_cache[cache_key]

    # 获取所有参数
    params = list(loss_fn.parameters())
    n_params = sum(p.numel() for p in params)

    # 重新计算损失和梯度以确保计算图正确
    loss = loss_fn()

    # 计算一阶导数
    first_grads = torch.autograd.grad(outputs=loss, inputs=params, create_graph=True, retain_graph=True)

    # 将一阶导数展平
    first_grad_flat = torch.cat([g.flatten() for g in first_grads])

    # 初始化Hessian矩阵
    hessian = torch.zeros(n_params, n_params, device=first_grad_flat.device, dtype=first_grad_flat.dtype)

    # 批量计算Hessian矩阵（优化版本）
    # 使用torch.autograd.functional.hessian如果可用，否则使用循环
    try:
        # 尝试使用更高效的批量计算方法
        param_vector = torch.cat([p.flatten() for p in params])

        # 定义一个函数，接受展平的参数向量并返回损失
        def loss_fn_flat(param_vec):
            # 重塑参数向量并更新模型参数
            idx = 0
            for p in params:
                param_size = p.numel()
                p.data = param_vec[idx : idx + param_size].reshape(p.shape)
                idx += param_size

            # 计算损失
            return loss_fn()

        # 使用torch.autograd.functional.hessian计算Hessian
        hessian_full = torch.autograd.functional.hessian(loss_fn_flat, param_vector)

        # 确保Hessian是对称的
        hessian = 0.5 * (hessian_full + hessian_full.T)

    except Exception as e:
        logger.warning(f"批量Hessian计算失败，回退到循环方法: {e}")

        # 回退到原始的循环方法
        for i in range(n_params):
            # 对第i个梯度分量求二阶导数
            second_grads = torch.autograd.grad(
                outputs=first_grad_flat[i], inputs=params, retain_graph=True, allow_unused=True
            )

            # 将二阶导数展平并存储到Hessian矩阵中
            start_idx = 0
            for param, second_grad in zip(params, second_grads):
                if second_grad is not None:
                    param_size = param.numel()
                    hessian[i, start_idx : start_idx + param_size] = second_grad.flatten()
                    start_idx += param_size

    # 确保Hessian是对称的
    hessian = 0.5 * (hessian + hessian.T)

    # 添加小的正则化项以确保数值稳定性
    eps = 1e-8
    hessian += eps * torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)

    # 缓存结果
    if cache_key is not None and hessian_cache is not None:
        hessian_cache[cache_key] = hessian

    return hessian


def fit(
    juno_syst,
    initial_params=None,
    n_iter=100,
    lr=1e-2,
    use_edm=False,
    edm_tolerance=1e-7,
    edm_method="approximate",
    edm_check_interval=10,  # 新增：EDM检查间隔
    use_scheduler=False,
    scheduler_type="plateau",
    scheduler_params=None,
    use_parameter_scaling=False,
    use_compile=False,
    compile_mode="reduce-overhead",
    compile_warmup=0,
    hessian_cache_interval=50,  # 新增：Hessian缓存更新间隔
    optimizer_type: str = "sgd",  # 新增：优化器类型 ('sgd','adam','adamw','adagrad')
    optimizer_params: Optional[Dict[str, Any]] = None,  # 新增：优化器参数
    enable_lr_schedule: bool = True,  # 新增：外部控制是否应用 learning_rate.json
    parameter_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
):
    """
    使用SGD进行拟合反应堆中微子参数，可选择使用EDM收敛判断

    Parameters
    ----------
    juno_syst : JunoSyst
        JUNO系统分析对象，包含观测数据和chi2计算方法
    initial_params : array-like, optional
        初始参数值，如果为None则使用juno_syst的默认参数
    n_iter : int, optional
        最大迭代次数，默认100
    lr : float, optional
        学习率（固定模式）或初始学习率（使用调度器时），默认1e-2
    use_edm : bool, optional
        是否使用EDM收敛判断，默认False
    edm_tolerance : float, optional
        EDM收敛阈值，默认1e-7
    edm_method : str, optional
        EDM计算方法，可选值：
        - 'approximate': 使用近似方法（快速，适合大型模型）
        - 'exact': 使用精确Hessian方法（准确，但计算成本高）
        - 'hybrid': 结合两种方法
        默认'approximate'
    edm_check_interval : int, optional
        EDM检查间隔，每多少次迭代检查一次EDM，默认10
    use_scheduler : bool, optional
        是否使用学习率调度器动态调整学习率，默认False
    scheduler_type : str, optional
        学习率调度器类型，可选值：
        - 'plateau': ReduceLROnPlateau，基于损失平台期降低学习率
        - 'exponential': ExponentialLR，指数衰减
        - 'cosine': CosineAnnealingLR，余弦退火
        - 'step': StepLR，固定步数衰减
        默认'plateau'
    scheduler_params : dict, optional
        调度器参数，不同调度器的参数：
        - plateau: {'factor': 0.5, 'patience': 50, 'threshold': 1e-4, 'min_lr': 1e-8}
        - exponential: {'gamma': 0.99}
        - cosine: {'T_max': 100, 'eta_min': 1e-8}
        - step: {'step_size': 100, 'gamma': 0.5}
        如果为None，使用默认参数
    use_parameter_scaling : bool, optional
        是否对振荡参数使用缩放，默认True
    hessian_cache_interval : int, optional
        Hessian缓存更新间隔，每多少次迭代更新一次Hessian缓存，默认50
    optimizer_type : str, optional
        优化器类型：'sgd' | 'adam' | 'adamw' | 'adagrad'，默认'sgd'
    optimizer_params : dict, optional
        传递给优化器的可选参数（会与lr合并），例如{'weight_decay':1e-3}
    enable_lr_schedule : bool, optional
        是否启用 learning_rate.json 提供的学习率序列；False 时即使文件存在也忽略
    parameter_limits : dict, optional
        参数名到 (lower, upper) 的映射，用于模仿 Minuit 的 m.limits 约束

    Returns
    -------
    tuple
        包含以下元素的元组：
        - x : list, 迭代步数历史
        - y : list, 损失函数值历史
        - t : list, 时间历史
        - lr : list, 学习率历史
        - reactor_loss : ReactorLoss, 损失函数对象
        - edm_values : list, EDM值历史

    Notes
    -----
    EDM (Estimated Distance to Minimum) 是iMinuit中用于判断拟合收敛的指标。
    - approximate方法：假设Hessian为对角矩阵，计算快速
    - exact方法：计算真实Hessian矩阵，更准确但计算成本高
    - hybrid方法：结合两种方法的优势

    参数缩放功能：
    - 所有拟合参数的缩放因子从parameter_errors.json文件读取
    - 使用参数误差作为缩放因子，提高数值稳定性
    """
    lr_schedule: Optional[List[float]] = None
    if enable_lr_schedule:
        try:
            with open("./fitter/data/Xuejq/learning_rate_juno.json", "r") as f:
                lr_data = json.load(f)

            if isinstance(lr_data, dict):
                lr_schedule = [float(lr_data[key]) for key in sorted(lr_data.keys(), key=lambda x: int(x))]
            elif isinstance(lr_data, list):
                lr_schedule = [float(value) for value in lr_data]
            else:
                logger.warning("learning_rate.json has unsupported structure; ignoring custom schedule")
                lr_schedule = None

            if lr_schedule:
                logger.info(f"Loaded learning rate schedule with {len(lr_schedule)} entries from learning_rate.json")
        except FileNotFoundError:
            logger.info("learning_rate.json not found; using default learning rate settings")
        except Exception as e:
            logger.warning(f"Failed to load learning_rate.json: {e}")
            lr_schedule = None

    reactor_loss = ReactorLoss(
        juno_syst, initial_params, use_parameter_scaling, use_compile=use_compile, compile_mode=compile_mode
    )

    param_limits = {}
    if parameter_limits:
        for name, bounds in parameter_limits.items():
            if name in reactor_loss.param_name_to_idx:
                lower, upper = bounds
                param_limits[reactor_loss.param_name_to_idx[name]] = (lower, upper)

    # 构建优化器
    opt_type = (optimizer_type or "sgd").lower()
    base_params = {"lr": lr}
    if optimizer_params:
        base_params.update(optimizer_params)
    if opt_type == "sgd":
        optimizer = torch.optim.SGD(reactor_loss.parameters(), **base_params)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(reactor_loss.parameters(), **base_params)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(reactor_loss.parameters(), **base_params)
    elif opt_type == "adagrad":
        optimizer = torch.optim.Adagrad(reactor_loss.parameters(), **base_params)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    params_log = {k: base_params.get(k) for k in ["lr", "weight_decay", "betas", "eps"] if k in base_params}
    logger.info(f"use_parameter_scaling: {use_parameter_scaling}, optimizer={opt_type}, params={params_log}")

    # 初始化Hessian缓存
    hessian_cache = {}
    last_hessian_update = -hessian_cache_interval  # 确保第一次迭代时更新Hessian

    if use_compile and reactor_loss.use_compile and compile_warmup > 0:
        logger.info(f"Running {compile_warmup} warm-up iterations for torch.compile...")
        t_warm = time.perf_counter()
        with torch.no_grad():
            for _ in range(compile_warmup):
                _ = reactor_loss()
        logger.info(f"torch.compile warm-up completed in {time.perf_counter() - t_warm:.4f}s")

    # Setup learning rate scheduler if requested
    scheduler = None
    if use_scheduler:
        if lr_schedule and enable_lr_schedule:
            logger.warning(
                "learning_rate.json schedule will override scheduler-controlled learning rates at each iteration"
            )
        # Default parameters for different schedulers
        default_params = {
            "plateau": {
                "factor": 0.5,
                "patience": 50,
                "threshold": 1e-4,
                "min_lr": 1e-8,
                "mode": "min",
                "verbose": False,
            },
            "exponential": {"gamma": 0.99},
            "cosine": {"T_max": 100, "eta_min": 1e-8},
            "step": {"step_size": 100, "gamma": 0.5},
        }

        # Use provided params or defaults
        params = scheduler_params if scheduler_params is not None else default_params.get(scheduler_type, {})
        # Create the appropriate scheduler
        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        elif scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        logger.info(f"Using {scheduler_type} learning rate scheduler with params: {params}")

    x = []
    y = []
    t = []
    lr = []
    edm_values = []  # 记录每次迭代的EDM值

    # 初始化EDM收敛检查器
    if use_edm:
        edm_checker = EDMConvergence(tolerance=edm_tolerance, up=1.0)  # up=1.0 for chi-square
        edm_history = []

    t0 = time.perf_counter()
    with torch.no_grad():
        logger.info(f"Initial (physical) params: {reactor_loss.get_fitted_params()}")
        if use_parameter_scaling:
            logger.info(f"Initial (scaled) params: {reactor_loss.get_scaled_params()}")
        logger.info(f"Initial chi2: {reactor_loss.get_chi2_value():.6f}")

    # 计时：累计反向传播时间与总迭代时间
    total_backward_time = 0.0
    total_iter_time = 0.0
    total_edm_time = 0.0  # 新增：EDM计算时间

    for i in range(n_iter):
        if lr_schedule and enable_lr_schedule:
            scheduled_lr = lr_schedule[i] if i < len(lr_schedule) else lr_schedule[-1]
            for param_group in optimizer.param_groups:
                param_group["lr"] = scheduled_lr
        else:
            scheduled_lr = optimizer.param_groups[0]["lr"]

        iter_start = time.perf_counter()
        optimizer.zero_grad()
        loss = reactor_loss()
        bw_start = time.perf_counter()
        loss.backward()
        total_backward_time += time.perf_counter() - bw_start

        # EDM收敛检查（类似iMinuit的方式）
        current_edm = None

        if use_edm and i % edm_check_interval == 0:  # 只在指定间隔检查EDM
            edm_start = time.perf_counter()

            # 收集所有参数的梯度
            gradients = []
            params = []
            for p in reactor_loss.parameters():
                if p.grad is not None:
                    gradients.append(p.grad.data.flatten())
                    params.append(p.data.flatten())
            logger.info(f"gradients: {gradients}")
            if gradients:
                gradient_vector = torch.cat(gradients)
                param_vector = torch.cat(params)
                current_lr = optimizer.param_groups[0]["lr"]

                # 检查梯度是否全为零或包含NaN/Inf
                if torch.any(torch.isnan(gradient_vector)) or torch.any(torch.isinf(gradient_vector)):
                    logger.warning(f"梯度包含NaN或Inf，跳过EDM计算 (iter {i})")
                    current_edm = None
                elif torch.allclose(gradient_vector, torch.zeros_like(gradient_vector), atol=1e-12):
                    logger.info(f"梯度接近零，EDM收敛 (iter {i})")
                    current_edm = torch.norm(gradient_vector).item() * current_lr / 2.0
                else:
                    # 根据选择的方法计算EDM
                    if edm_method == "approximate":
                        # 使用近似方法计算EDM
                        edm_primary = edm_checker.compute_edm_approximate(gradient_vector, param_vector, current_lr)

                    elif edm_method == "exact":
                        # 使用精确Hessian方法计算EDM
                        try:
                            # 使用缓存机制
                            cache_key = f"iter_{i}"
                            if i - last_hessian_update >= hessian_cache_interval:
                                hessian = compute_hessian(reactor_loss, cache_key, hessian_cache)
                                last_hessian_update = i
                            else:
                                # 尝试从缓存获取
                                if cache_key in hessian_cache:
                                    hessian = hessian_cache[cache_key]
                                else:
                                    hessian = compute_hessian(reactor_loss, cache_key, hessian_cache)
                                    last_hessian_update = i

                            edm_primary = edm_checker.compute_edm(gradient_vector, hessian=hessian)
                        except Exception as e:
                            logger.warning(f"精确EDM计算失败，回退到近似方法: {e}")
                            edm_primary = edm_checker.compute_edm_approximate(gradient_vector, param_vector, current_lr)

                    elif edm_method == "hybrid":
                        # 混合方法：结合近似和精确方法
                        edm_approx = edm_checker.compute_edm_approximate(gradient_vector, param_vector, current_lr)

                        try:
                            # 使用缓存机制
                            cache_key = f"iter_{i}_hybrid"
                            if i - last_hessian_update >= hessian_cache_interval:
                                hessian = compute_hessian(reactor_loss, cache_key, hessian_cache)
                                last_hessian_update = i
                            else:
                                # 尝试从缓存获取
                                if cache_key in hessian_cache:
                                    hessian = hessian_cache[cache_key]
                                else:
                                    hessian = compute_hessian(reactor_loss, cache_key, hessian_cache)
                                    last_hessian_update = i

                            edm_exact = edm_checker.compute_edm(gradient_vector, hessian=hessian)
                            # 给精确方法更高权重
                            edm_primary = 0.7 * edm_exact + 0.3 * edm_approx
                        except Exception as e:
                            logger.warning(f"精确EDM计算失败，使用近似方法: {e}")
                            edm_primary = edm_approx

                    else:
                        raise ValueError(f"不支持的EDM方法: {edm_method}. 支持的方法: 'approximate', 'exact', 'hybrid'")

                    # 使用损失历史来改进EDM估计（所有方法都适用）
                    if len(y) >= 3:
                        try:
                            edm_from_loss = edm_checker.compute_edm_from_loss_history(y[-3:])
                            edm_loss_tensor = (
                                torch.tensor(edm_from_loss)
                                if not isinstance(edm_from_loss, torch.Tensor)
                                else edm_from_loss
                            )
                            if not (torch.isnan(edm_loss_tensor) or torch.isinf(edm_loss_tensor)):
                                # 综合主要方法和损失历史方法
                                current_edm = 0.8 * edm_primary + 0.2 * edm_from_loss
                            else:
                                current_edm = edm_primary
                        except Exception:
                            current_edm = edm_primary
                    else:
                        current_edm = edm_primary

                if current_edm is not None:
                    edm_history.append(current_edm)

                # 检查是否收敛
                if current_edm is not None:
                    converged, status = edm_checker.check_convergence(current_edm)

                    # 减少日志输出频率
                    if i % edm_check_interval == 0 or converged:
                        logger.info(
                            f"Iter {i}: Loss={loss.item():.6f}, EDM={current_edm:.6e} (method={edm_method}), {status}"
                        )

                    if converged:
                        logger.info(f"✓ EDM收敛于第 {i} 次迭代 (method={edm_method}): {status}")
                        # 记录最后一次迭代的数据
                        x.append(i)
                        t.append(time.perf_counter() - t0)
                        y.append(loss.item())
                        lr.append(optimizer.param_groups[0]["lr"])
                        edm_values.append(current_edm)
                        break

            total_edm_time += time.perf_counter() - edm_start

        # 梯度裁剪防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(reactor_loss.parameters(), max_norm=1.0)

        optimizer.step()

        if param_limits:
            with torch.no_grad():
                for idx, (lower, upper) in param_limits.items():
                    if lower is not None:
                        reactor_loss.params[idx] = torch.maximum(
                            reactor_loss.params[idx],
                            torch.tensor(lower, dtype=reactor_loss.params.dtype, device=reactor_loss.params.device),
                        )
                    if upper is not None:
                        reactor_loss.params[idx] = torch.minimum(
                            reactor_loss.params[idx],
                            torch.tensor(upper, dtype=reactor_loss.params.dtype, device=reactor_loss.params.device),
                        )

        # 输出物理空间的参数值（还原后的参数）- 减少输出频率
        if i % edm_check_interval == 0:
            with torch.no_grad():
                physical_params = reactor_loss._scale_params(reactor_loss.params.detach(), forward=True)
            logger.info(f"Physical params: {physical_params}")

        # Update learning rate scheduler if enabled
        if scheduler is not None and not (lr_schedule and enable_lr_schedule):
            if scheduler_type == "plateau":
                # ReduceLROnPlateau needs the loss value
                scheduler.step(loss.item())
            else:
                # Other schedulers just need to be stepped
                scheduler.step()

        x.append(i)
        t.append(time.perf_counter() - t0)
        y.append(loss.item())
        lr.append(optimizer.param_groups[0]["lr"])
        total_iter_time += time.perf_counter() - iter_start

        # 记录EDM值（如果计算了的话）
        if current_edm is not None:
            edm_values.append(current_edm)
        else:
            edm_values.append(None)  # 占位符，表示未计算EDM

        # 保留原有的简单收敛判断（作为备用）
        if not use_edm and len(y) >= 2 and np.abs(y[-1] - y[-2]) < 1e-7:
            break

    if use_edm and edm_history:
        logger.info(f"最终EDM: {edm_history[-1]:.6e} (method={edm_method})")

    # 输出时间占比
    total_iter_time = max(total_iter_time, 1e-12)
    backward_frac = total_backward_time / total_iter_time
    edm_frac = total_edm_time / total_iter_time
    logger.info(
        f"时间统计: backward={total_backward_time:.4f}s ({backward_frac * 100:.2f}%), "
        f"EDM计算={total_edm_time:.4f}s ({edm_frac * 100:.2f}%), "
        f"total={total_iter_time:.4f}s"
    )

    return x, y, t, lr, reactor_loss, edm_values


def fit_lbfgs(
    juno_syst,
    initial_params=None,
    max_iter: int = 200,
    lr: float = 1.0,
    history_size: int = 10,
    line_search_fn: str = "strong_wolfe",
    edm_tolerance: float = 1e-5,
    edm_method: str = "lbfgs",
    use_parameter_scaling: bool = True,
    parameter_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Tuple[list, list, list, list, ReactorLoss, list]:
    """
    L-BFGS fitting with EDM convergence criterion.

    Parameters
    ----------
    juno_syst : JunoSyst
        JUNO system object with observed data loaded.
    initial_params : array-like, optional
        Initial parameter values. None uses defaults from juno_syst.
    max_iter : int
        Maximum number of outer L-BFGS steps.
    lr : float
        L-BFGS learning rate (default 1.0; line search handles step size).
    history_size : int
        Number of curvature pairs stored by L-BFGS.
    line_search_fn : str
        Line search method ('strong_wolfe' recommended).
    edm_tolerance : float
        Stop when EDM < edm_tolerance.
    edm_method : str
        EDM calculation: 'lbfgs' (two-loop recursion), 'gradient_norm',
        or 'exact' (full Hessian).
    use_parameter_scaling : bool
        Whether to scale parameters for numerical stability.
    parameter_limits : dict, optional
        Parameter name to (lower, upper) bounds.

    Returns
    -------
    tuple : (x, y, t, lr_list, reactor_loss, edm_values)
        Same 6-tuple as fit() for compatibility.
    """
    reactor_loss = ReactorLoss(juno_syst, initial_params, use_parameter_scaling)

    # Build parameter bounds index
    param_limits = {}
    if parameter_limits:
        for name, bounds in parameter_limits.items():
            if name in reactor_loss.param_name_to_idx:
                param_limits[reactor_loss.param_name_to_idx[name]] = bounds

    optimizer = torch.optim.LBFGS(
        reactor_loss.parameters(),
        lr=lr,
        max_iter=1,  # one L-BFGS update per outer step for proper tracking
        history_size=history_size,
        line_search_fn=line_search_fn,
        tolerance_grad=0,
        tolerance_change=0,
    )

    def closure():
        optimizer.zero_grad()
        loss = reactor_loss()
        loss.backward()
        return loss

    with torch.no_grad():
        logger.info(f"[L-BFGS] Initial params: {reactor_loss.get_fitted_params()}")
        logger.info(f"[L-BFGS] Initial chi2: {reactor_loss.get_chi2_value():.6f}")

    edm_checker = EDMConvergence(tolerance=edm_tolerance)
    x, y, t, lr_list, edm_values = [], [], [], [], []
    t0 = time.perf_counter()

    for i in range(max_iter):
        optimizer.step(closure)

        # Re-evaluate to get current loss and gradient at updated params
        optimizer.zero_grad()
        loss = reactor_loss()
        loss.backward()
        loss_val = loss.item()

        # Enforce parameter limits
        if param_limits:
            with torch.no_grad():
                for idx, (lower, upper) in param_limits.items():
                    if lower is not None:
                        reactor_loss.params[idx] = torch.maximum(
                            reactor_loss.params[idx],
                            torch.tensor(lower, dtype=reactor_loss.params.dtype),
                        )
                    if upper is not None:
                        reactor_loss.params[idx] = torch.minimum(
                            reactor_loss.params[idx],
                            torch.tensor(upper, dtype=reactor_loss.params.dtype),
                        )

        # Record history
        x.append(i)
        y.append(loss_val)
        t.append(time.perf_counter() - t0)
        lr_list.append(lr)

        # Compute EDM
        grad_flat = reactor_loss.params.grad.detach().flatten()
        if edm_method == "lbfgs":
            edm = compute_edm_from_lbfgs_state(grad_flat, optimizer)
        elif edm_method == "gradient_norm":
            edm = compute_edm_gradient_norm(grad_flat)
        elif edm_method == "approximate":
            edm = edm_checker.compute_edm_approximate(grad_flat, reactor_loss.params, lr)
        elif edm_method == "exact":
            hessian = compute_hessian(reactor_loss)
            edm = edm_checker.compute_edm(grad_flat, hessian=hessian)
        else:
            raise ValueError(f"Unknown edm_method: {edm_method}")
        edm_values.append(edm)

        logger.info(f"[L-BFGS] Step {i}: loss={loss_val:.6e}, EDM={edm:.6e}")

        if edm < edm_tolerance:
            logger.info(f"[L-BFGS] Converged at step {i}: EDM={edm:.6e} < {edm_tolerance:.1e}")
            break

    with torch.no_grad():
        logger.info(f"[L-BFGS] Final params: {reactor_loss.get_fitted_params()}")
        logger.info(f"[L-BFGS] Final chi2: {reactor_loss.get_chi2_value():.6f}")
        logger.info(f"[L-BFGS] Total time: {time.perf_counter() - t0:.4f}s")

    return x, y, t, lr_list, reactor_loss, edm_values


def gradient_step0_report(juno_syst, lr=1e-7, use_parameter_scaling=False, eps_rel=1e-6):
    """
    计算第0次迭代的梯度（两种方法）并输出：
    - 初值（物理空间参数）
    - 梯度（backward自动求导 与 手动中心差分）
    - 一步SGD后的参数值（物理空间）

    Parameters
    ----------
    juno_syst : JunoSyst
        JUNO系统对象，需已加载或生成观测数据
    lr : float
        SGD学习率，用于执行单步更新
    use_parameter_scaling : bool
        是否启用参数缩放（与 ReactorLoss 保持一致）
    eps_rel : float
        计算数值梯度的相对步长

    Returns
    -------
    dict
        包含初值、两种梯度、更新后值的字典
    """
    device = juno_syst.fit_para_init.device
    dtype = juno_syst.fit_para_init.dtype

    # 构造损失对象（不改变 juno_syst 内部状态）
    reactor_loss = ReactorLoss(juno_syst, initial_params=None, use_parameter_scaling=use_parameter_scaling)

    # 初始参数（物理空间）
    with torch.no_grad():
        physical_init = reactor_loss._scale_params(reactor_loss.params.detach(), forward=True).clone()

    # 方法一：backward 自动求导（得到的是对缩放空间的梯度，需要换算到物理空间）
    reactor_loss.zero_grad()
    loss0 = reactor_loss()
    loss0.backward()
    grad_scaled = reactor_loss.params.grad.detach().clone()

    # 组装缩放因子（d physical / d scaled）。未缩放的参数因子为1。
    scale_factors = torch.ones_like(grad_scaled)
    if use_parameter_scaling:
        for name, factor in reactor_loss.param_scales.items():
            if name in reactor_loss.param_name_to_idx:
                idx = reactor_loss.param_name_to_idx[name]
                scale_factors[idx] = factor
    # dchi2/dphysical = dchi2/dscaled / (d physical / d scaled)
    grad_physical_backward = grad_scaled / scale_factors

    # 方法二：中心差分数值梯度（手动）在物理空间上进行
    n_params = len(juno_syst.fit_para_names)
    grad_physical_fd = torch.zeros(n_params, device=device, dtype=dtype)

    def chi2_at(physical_vec: torch.Tensor) -> float:
        return juno_syst.chi2(physical_vec).item()

    for i in range(n_params):
        base = physical_init[i].item()
        h = eps_rel * max(1.0, abs(base))
        vec_plus = physical_init.clone()
        vec_plus[i] = base + h
        vec_minus = physical_init.clone()
        vec_minus[i] = base - h
        f_plus = chi2_at(vec_plus)
        f_minus = chi2_at(vec_minus)
        grad_physical_fd[i] = (f_plus - f_minus) / (2.0 * h)

    # 单步SGD更新（使用backward得到的梯度）
    optimizer = torch.optim.SGD([reactor_loss.params], lr=lr)
    reactor_loss.zero_grad()
    loss0 = reactor_loss()
    loss0.backward()
    optimizer.step()

    with torch.no_grad():
        physical_updated = reactor_loss._scale_params(reactor_loss.params.detach(), forward=True).clone()

    # 打印输出
    print("Initial parameters (physical), Grad at step 0 (backward & finite-diff), and Updated (physical):")
    for i, name in enumerate(juno_syst.fit_para_names):
        init_val = physical_init[i].item()
        g_bwd = grad_physical_backward[i].item()
        g_fd = grad_physical_fd[i].item()
        upd_val = physical_updated[i].item()
        print(f"{name}: init={init_val:.10e}, grad_bwd={g_bwd:.10e}, grad_fd={g_fd:.10e}, updated={upd_val:.10e}")

    return {
        "names": list(juno_syst.fit_para_names),
        "initial_physical": physical_init.cpu().numpy(),
        "grad_backward_physical": grad_physical_backward.cpu().numpy(),
        "grad_finite_diff_physical": grad_physical_fd.cpu().numpy(),
        "updated_physical": physical_updated.cpu().numpy(),
    }


def scan_lr_schedule(
    juno_syst,
    initial_params: Optional[Sequence[float]] = None,
    candidate_lrs: Optional[Sequence[float]] = None,
    n_steps: int = 40,
    use_parameter_scaling: bool = True,
    plot_dir: str = "lr_scans",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Per-iteration learning-rate scan that selects the LR with the largest loss decrease
    for each step. At each iteration k:
      1) Compute loss and gradient at current parameters
      2) For each candidate learning rate, simulate one SGD step and measure loss change
      3) Choose the LR that maximizes (loss_before - loss_after) and commit the update
      4) Record chosen LR and loss change; save a plot (x: lr, y: loss decrease)

    Parameters
    ----------
    juno_syst : JunoSyst
        JUNO system object with chi2 implemented
    initial_params : Sequence[float], optional
        Initial physical-space parameters. If None, uses juno_syst.fit_para_init
    candidate_lrs : Sequence[float], optional
        Learning rates to scan each step. If None, uses a default geometric range
    n_steps : int
        Number of outer iterations (scans)
    use_parameter_scaling : bool
        Whether to enable parameter scaling inside ReactorLoss
    plot_dir : str
        Directory to save per-step plots (lr vs loss decrease)
    verbose : bool
        If True, logs per-step summary

    Returns
    -------
    dict
        {
          "chosen_lrs": List[float],
          "loss_changes": List[float],  # loss_before - loss_after per step
          "loss_history": List[float],  # loss after each committed step
          "per_step_scan": List[Tuple[List[float], List[float]]],  # (lrs, deltas)
        }
    """
    import matplotlib.pyplot as plt

    if candidate_lrs is None:
        candidate_lrs = np.geomspace(0.1, 1, 20)
        candidate_lrs = torch.tensor(candidate_lrs)

    os.makedirs(plot_dir, exist_ok=True)

    reactor_loss = ReactorLoss(
        juno_syst=juno_syst,
        initial_params=initial_params,
        use_parameter_scaling=use_parameter_scaling,
    )

    chosen_lrs: List[float] = []
    loss_changes: List[float] = []
    loss_history: List[float] = []
    per_step_scan: List[Tuple[List[float], List[float]]] = []

    with torch.no_grad():
        current_loss = reactor_loss().item()
    loss_history.append(current_loss)

    for step_index in range(n_steps):
        reactor_loss.zero_grad()
        loss_before = reactor_loss()
        loss_before_value = loss_before.item()
        loss_before.backward()
        # logger.info(f"Step {step_index:03d}: loss_before={loss_before_value:.12e}")
        if reactor_loss.params.grad is None:
            raise RuntimeError("Gradient is None; cannot perform LR scan.")

        gradient_scaled = reactor_loss.params.grad.detach().clone()
        base_params = reactor_loss.params.detach().clone()
        base_params_physical = reactor_loss._scale_params(base_params, forward=True).cpu().numpy()
        # logger.info(f"Step {step_index:03d}: base_params={base_params_physical}")
        lrs_this_step: List[float] = []
        deltas_this_step: List[float] = []

        for lr_value in candidate_lrs:
            proposed_params = base_params - lr_value * gradient_scaled
            with torch.no_grad():
                reactor_loss.params.copy_(proposed_params)
                loss_after_value = reactor_loss().item()
                reactor_loss.params.copy_(base_params)

            delta = loss_before_value - loss_after_value
            lrs_this_step.append(lr_value)
            deltas_this_step.append(delta)

        per_step_scan.append((lrs_this_step, deltas_this_step))

        best_index = int(torch.tensor(deltas_this_step).argmax().item())
        best_lr = lrs_this_step[best_index]
        best_delta = deltas_this_step[best_index]

        with torch.no_grad():
            committed_params = base_params - best_lr * gradient_scaled
            reactor_loss.params.copy_(committed_params)
            committed_loss = reactor_loss().item()

        chosen_lrs.append(best_lr)
        loss_changes.append(best_delta)
        loss_history.append(committed_loss)

        if verbose:
            logger.info(
                f"Step {step_index:03d}: loss_before={loss_before_value:.16e}, "
                f"best_lr={best_lr:.3e}, delta={best_delta:.6e}, loss_after={committed_loss:.16e}"
            )

        plt.figure(figsize=(8, 5))
        plt.plot(lrs_this_step, deltas_this_step, "o-", color="#1f77b4", linewidth=2)
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss decrease (loss_before - loss_after)")
        plt.title(f"LR scan at step {step_index}")
        plt.grid(True, alpha=0.3)
        plt.axvline(best_lr, color="#d62728", linestyle="--", label=f"best lr={best_lr:.3e}")
        plt.legend()
        out_path = os.path.join(plot_dir, f"lr_scan_step_{step_index:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "chosen_lrs": chosen_lrs,
        "loss_changes": loss_changes,
        "loss_history": loss_history,
        "per_step_scan": per_step_scan,
    }
