from iminuit import Minuit
from src.fitter.analysis.fitter import Fitter
import psutil
import os
import time
import json
import torch
from src.fitter.config import GlobalConfig as gcfg

torch.set_num_interop_threads(1)
torch.set_num_threads(1)
gcfg.device = "cuda"
start_time = time.perf_counter()
process = psutil.Process(os.getpid())
initial_cpu = process.cpu_times()
gseed = 1  # 0 for Asimov >0 for toy
year = 1  # 1年统计量
# gcfg.use_poisson = True  # 是否使用泊松分布
# if gseed < 0:
#     print("Seed must be greater than 0")
#     exit()

fitter = Fitter(year)
fitter.get_obs_spectrum(gseed)

# fitter.initial_params_inverted对应逆序拟合，fitter.initial_params_normal对应正序拟合
m = Minuit(fitter.chi2, fitter.NO_params_NO, name=fitter.names)

m.tol = 0.01
# m.fixed["dmsq31"] = True
results = m.migrad()

# 保存拟合误差到 JSON 文件
errors_dict = {}
for name in m.parameters:
    errors_dict[name] = round(float(m.errors[name]), 6)

output_dir = "fitter/data/xuejq"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "scale_factor_juno.json")
with open(output_path, "w") as f:
    json.dump(errors_dict, f, indent=2)
print(f"Scale factor saved to {output_path}")