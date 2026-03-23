import torch.nn as nn
import torch
import json
import os
from loguru import logger
from ..config import GlobalConfig as gcfg

torch.set_default_dtype(torch.float64)


class ReactorLoss(nn.Module):
    """
    反应堆中微子振荡参数拟合的Loss类
    使用JunoSyst的chi2方法计算损失值，并对所有拟合参数进行缩放
    缩放因子从parameter_errors.json文件中读取，使用参数误差作为缩放因子
    """

    def __init__(
        self,
        juno_syst,
        initial_params=None,
        use_parameter_scaling=True,
        use_compile=False,
        compile_mode="reduce-overhead",
    ):
        """
        Parameters
        ----------
        juno_syst : JunoSyst
            JUNO系统分析对象
        initial_params : array-like, optional
            初始参数值，如果为None则使用默认的fit_para_init
        use_parameter_scaling : bool, optional
            是否对所有拟合参数使用缩放，默认True
            缩放因子从parameter_errors.json文件读取
        """
        super(ReactorLoss, self).__init__()

        self.juno_syst = juno_syst
        self.use_parameter_scaling = use_parameter_scaling
        self.use_compile = use_compile
        self.compile_mode = compile_mode

        # 从parameter_errors.json读取参数缩放因子
        self.param_scales = self._load_parameter_scales()

        # 参数名称到索引的映射
        self.param_name_to_idx = {}
        for i, name in enumerate(self.juno_syst.fit_para_names):
            self.param_name_to_idx[name] = i

        # 初始化参数
        if initial_params is None:
            raw_params = self.juno_syst.fit_para_init.clone()
        else:
            raw_params = torch.tensor(initial_params)

        # 如果使用缩放，则对初始参数进行缩放
        if self.use_parameter_scaling:
            scaled_params = self._scale_params(raw_params, forward=False)  # 缩放参数用于训练
        else:
            scaled_params = raw_params

        self.params = nn.Parameter(scaled_params)

        # 可选：使用 torch.compile 加速前向
        if self.use_compile:
            try:
                if hasattr(torch, "compile"):
                    self.forward = torch.compile(self.forward, mode=self.compile_mode)
                else:
                    logger.warning("torch.compile 不可用（PyTorch<2.0），改用 eager 模式")
                    self.use_compile = False
            except Exception as e:
                logger.warning(f"torch.compile 编译失败，回退到 eager 模式: {e}")
                self.use_compile = False

    def _load_parameter_scales(self):
        """
        从parameter_errors.json文件读取参数误差值作为缩放因子

        Returns
        -------
        dict
            参数名称到缩放因子的映射
        """
        json_path = gcfg.scale_factor_path

        # 如果JSON文件不存在，使用默认的缩放因子
        if not os.path.exists(json_path):
            logger.warning(f"Parameter errors file {json_path} not found, using default scaling factors")
            return {"dmsq31": 1.8e-05, "sinsq12": 0.006, "dmsq21": 8.5e-7, "sinsq13": 0.7e-3}

        try:
            with open(json_path, "r") as f:
                errors_dict = json.load(f)

            logger.info(f"Loaded parameter scaling factors from {json_path}")
            logger.info(f"Found scaling factors for {len(errors_dict)} parameters")

            return errors_dict

        except Exception as e:
            logger.error(f"Failed to load parameter errors from {json_path}: {e}")
            logger.info("Using default scaling factors")
            return {"dmsq31": 1.8e-05, "sinsq12": 0.006, "dmsq21": 8.5e-7, "sinsq13": 0.7e-3}

    def _scale_params(self, params, forward=True):
        """
        参数缩放函数

        Parameters
        ----------
        params : torch.Tensor
            参数向量
        forward : bool
            True: 从缩放空间转换到物理空间 (scaled -> physical)
            False: 从物理空间转换到缩放空间 (physical -> scaled)
        """
        if not self.use_parameter_scaling:
            return params

        scaled_params = params.clone()

        # 对需要缩放的参数进行处理
        for param_name, scale_factor in self.param_scales.items():
            if param_name in self.param_name_to_idx:
                idx = self.param_name_to_idx[param_name]
                if forward:
                    # 缩放空间 -> 物理空间
                    scaled_params[idx] = params[idx] * scale_factor
                else:
                    # 物理空间 -> 缩放空间
                    scaled_params[idx] = params[idx] / scale_factor

        return scaled_params

    def forward(self):
        """
        计算损失函数值（卡方值）

        Returns
        -------
        torch.Tensor
            卡方损失值
        """
        # 将缩放空间的参数转换为物理空间
        physical_params = self._scale_params(self.params, forward=True)

        # 使用JunoSyst计算chi2
        return self.juno_syst.chi2(physical_params)

    def get_fitted_params(self):
        """
        获取拟合后的参数值（物理空间）

        Returns
        -------
        dict
            参数名称和值的字典
        """
        with torch.no_grad():
            # 转换为物理空间参数
            physical_params = self._scale_params(self.params, forward=True)

            param_dict = {}
            for i, name in enumerate(self.juno_syst.fit_para_names):
                param_dict[name] = physical_params[i].item()
        return param_dict

    def get_oscillation_params(self):
        """
        获取四个关键振荡参数（物理空间）

        Returns
        -------
        dict
            包含四个振荡参数的字典
        """
        fitted_params = self.get_fitted_params()
        osc_params = {}

        # 提取四个关键振荡参数
        osc_params["dmsq31"] = fitted_params.get("dmsq31", 0.0)
        osc_params["sinsq12"] = fitted_params.get("sinsq12", 0.0)
        osc_params["dmsq21"] = fitted_params.get("dmsq21", 0.0)
        osc_params["sinsq13"] = fitted_params.get("sinsq13", 0.0)

        return osc_params

    def get_scaled_params(self):
        """
        获取缩放空间的参数值（用于内部调试）

        Returns
        -------
        dict
            缩放空间的参数字典
        """
        with torch.no_grad():
            param_dict = {}
            for i, name in enumerate(self.juno_syst.fit_para_names):
                param_dict[name] = self.params[i].item()
        return param_dict

    def get_chi2_value(self):
        """获取当前参数下的卡方值"""
        with torch.no_grad():
            return self.forward().item()

    def get_parameter_scales(self):
        """获取参数缩放因子信息"""
        return self.param_scales.copy()


# 为了向后兼容，保留Loss类的别名
class Loss(ReactorLoss):
    """向后兼容的Loss类别名"""

    def __init__(self, juno_syst, initial_params=None, use_parameter_scaling=True):
        super(Loss, self).__init__(juno_syst, initial_params, use_parameter_scaling)
