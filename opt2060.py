import numpy as np
import os
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from joblib import Parallel, delayed


# 定义自定义采样类
class CustomSampling(Sampling):
    def __init__(self, M, T, alpha, initial_structure, xl, xu):
        super().__init__()
        self.M = M
        self.T = T
        self.alpha = alpha
        self.initial_structure = initial_structure
        self.xl = xl
        self.xu = xu

    def _do(self, problem, n_samples, **kwargs):
        samples = np.zeros((n_samples, problem.n_var), dtype=np.float64)

        for k in range(n_samples):
            sample = np.zeros(problem.n_var, dtype=np.float64)
            # 在每个变量的上下限内随机生成个体
            for i in range(problem.n_var):
                sample[i] = np.float64(np.random.uniform(self.xl[i], self.xu[i]))

            samples[k, :] = sample  # 赋值给样本数组

            # 评估个体是否满足约束
            out = {"F": np.zeros(problem.n_obj), "G": np.zeros(problem.n_constr)}  # 初始化out字典
            problem._evaluate(sample, out=out)

            # 检查out["G"]是否为有效的数组
            if isinstance(out["G"], np.ndarray) and not np.isnan(out["G"]).any():
                # 计算违反的约束数量
                constraint_violations = np.count_nonzero(out["G"] > 0)
                # 判断是否违反了太多约束
                if constraint_violations > 0.5 * problem.n_constr:
                    print(f"Sample {k} violates too many constraints ({constraint_violations}), regenerating...")
                else:
                    samples[k, :] = sample
                    k += 1  # 仅当样本有效时才增加计数
            else:
                print(f"Invalid constraints evaluation for sample {k}, regenerating...")

        return samples


# 定义优化问题类
class IntegratedOptimizationProblem(ElementwiseProblem):
    def __init__(self, M, T, a, alpha, initial_gdp, initial_structure, Ec, Ues, Les, mu,
                 initial_energy_consumption, Lp, Up, initial_ief, ief, Cp_ratio, Ch_ratio, phi_c, Lci, Uci, CE_lim,
                 se, pe, ne, phi_s, phi_p, phi_n, demand_ratio, residential_energy_ratio, residential_se_ratio,
                 residential_pe_ratio,
                 residential_ne_ratio, initial_pe_se, initial_pe_pe, initial_pe_ne, energy_efficiency):
        super().__init__(
            n_var=M * T,  # 总变量数，包括经济产出变量
            n_obj=3,  # 三个目标函数：最大化GDP，最小化碳排放，最小化污染物排放
            n_constr=(6 * M + 10) * T + 4,  # 约束总数（根据去掉后的约束进行调整）
            xl=np.zeros(M * T),  # 变量下限，所有变量的最小值都设为0
            xu=np.full(M * T, 1000)  # 变量上限，所有变量的最大值都设为1000
        )
        self.M = M
        self.T = T
        self.a = a
        self.alpha = alpha
        self.initial_gdp = initial_gdp
        self.initial_structure = initial_structure
        self.Ec = Ec
        self.Ues = Ues
        self.Les = Les
        self.mu = mu
        self.initial_energy_consumption = initial_energy_consumption
        self.Lp = Lp
        self.Up = Up
        self.ief = ief
        self.Cp_ratio = Cp_ratio
        self.Ch_ratio = Ch_ratio
        self.phi_c = phi_c
        self.Lci = Lci
        self.Uci = Uci
        self.CE_lim = CE_lim
        self.se = se
        self.pe = pe
        self.ne = ne
        self.phi_s = phi_s
        self.phi_p = phi_p
        self.phi_n = phi_n
        self.demand_ratio = demand_ratio
        self.residential_energy_ratio = residential_energy_ratio
        self.residential_se_ratio = residential_se_ratio
        self.residential_pe_ratio = residential_pe_ratio
        self.residential_ne_ratio = residential_ne_ratio
        self.initial_pe_se = initial_pe_se
        self.initial_pe_pe = initial_pe_pe
        self.initial_pe_ne = initial_pe_ne
        self.energy_efficiency = energy_efficiency
        self.initial_ief = initial_ief  # 确保初始的碳排放系数

        # 设置变量的上下限
        self.set_variable_bounds()
        # 初始化自适应罚函数的参数
        self.penalty_weight = 5000  # 罚函数权重初始化为1.0

    def set_variable_bounds(self):
        """设置变量的上下限"""
        # 设置经济产出的上下限
        for t in range(self.T):
            for m in range(self.M):
                gdp_base_low = self.initial_structure[m] * np.prod([(1 + self.alpha[j]) for j in range(t + 1)])
                gdp_base_high = self.initial_structure[m] * np.prod([(1 + 1.5 * self.alpha[j]) for j in range(t + 1)])
                self.xl[t * self.M + m] = 0.95 ** (t + 1) * gdp_base_low / (1 - np.sum(self.a, axis=0))[m]
                self.xu[t * self.M + m] = 1.05 ** (t + 1) * gdp_base_high / (1 - np.sum(self.a, axis=0))[m]

    def add_economic_constraints(self, x_econ, t, g, constraint_index):
        """添加经济模块的约束"""

        # 计算 GDP
        GDP_t = np.sum(x_econ[t, :] * (1 - np.sum(self.a, axis=0)))
        if t == 0:
            previous_gdp = self.initial_gdp
        else:
            previous_gdp = np.sum(x_econ[t - 1, :] * (1 - np.sum(self.a, axis=0)))

        # GDP 增长率约束

        g[constraint_index] = ((1 + self.alpha[t]) * previous_gdp - GDP_t) / ((1 + self.alpha[t]) * previous_gdp)
        constraint_index += 1
        # 经济结构稳定性约束
        if t == 0:
            previous_structure = self.initial_structure / np.sum(self.initial_structure)
        else:
            previous_structure = x_econ[t - 1, :] * (1 - np.sum(self.a, axis=0)) / np.sum(
                (x_econ[t - 1, :]) * (1 - np.sum(self.a, axis=0)))
        current_structure = x_econ[t, :] * (1 - np.sum(self.a, axis=0)) / np.sum(
            (x_econ[t, :]) * (1 - np.sum(self.a, axis=0)))
        for i in range(self.M):
            ratio_previous = previous_structure[i] / np.sum(previous_structure)
            ratio_current = current_structure[i] / np.sum(current_structure)
            g[constraint_index] = (1 + self.Lp) - (ratio_current / ratio_previous)
            constraint_index += 1
            g[constraint_index] = (ratio_current / ratio_previous) - (1 + self.Up)
            constraint_index += 1

        return constraint_index

    def add_energy_constraints(self, x_econ, t, g, constraint_index):
        """添加能源模块的约束"""
        # 计算总能耗
        energy_total_consumption = np.sum(self.energy_efficiency[t, :] * x_econ[t, :] * (1 - np.sum(self.a, axis=0)))
        residential_energy_type_consumption = self.residential_energy_ratio * energy_total_consumption

        # 总能耗约束
        g[constraint_index] = ((energy_total_consumption + residential_energy_type_consumption) - 1.5 * self.Ec[t]) / (
                    1.5 * self.Ec[t])
        constraint_index += 1
        g[constraint_index] = (0.5 * self.Ec[t] - (energy_total_consumption + residential_energy_type_consumption)) / (
                    0.50 * self.Ec[t])
        constraint_index += 1

        # # 各产业能耗增长率约束
        # for i in range(self.M):
        #     if t == 0:
        #         previous_total = np.sum(self.initial_energy_consumption[i])
        #         previous_energy_total_consumption = np.sum(self.initial_energy_consumption)  # 上一年的能源总消费为初始消费
        #     else:
        #         previous_total = np.sum(
        #             self.energy_efficiency[t - 1, i] * x_econ[t - 1, i] * (1 - np.sum(self.a[:, i])))
        #         previous_energy_total_consumption = np.sum(
        #             self.energy_efficiency[t - 1, :] * x_econ[t - 1, :] * (1 - np.sum(self.a, axis=0)))  # 上一年的能源总消费
        #
        #     current_total = np.sum(self.energy_efficiency[t, i] * x_econ[t, i] * (1 - np.sum(self.a[:, i])))
        #     current_energy_total_consumption = np.sum(
        #         self.energy_efficiency[t, :] * x_econ[t, :] * (1 - np.sum(self.a, axis=0)))  # 当年的能源总消费
        #
        #     energy_growth_rate = current_energy_total_consumption / previous_energy_total_consumption
        #
        #     growth_rate = current_total / previous_total
        #     g[constraint_index] = ((1 + self.Les) * energy_growth_rate - growth_rate)
        #     constraint_index += 1
        #     g[constraint_index] = growth_rate - (1 + self.Ues) * energy_growth_rate
        #     constraint_index += 1
        return constraint_index

    def add_carbon_constraints(self, x_econ, t, g, constraint_index):
        """添加碳排放约束"""
        if t == 0:
            previous_ce = np.sum(self.initial_energy_consumption * self.initial_ief) * (
                        1 + self.Cp_ratio + self.Ch_ratio)
        else:
            previous_ce = np.sum(
                self.energy_efficiency[t - 1, :] * x_econ[t - 1, :] * (1 - np.sum(self.a, axis=0)) * self.ief[t - 1,
                                                                                                     :]) * (
                                      1 + self.Cp_ratio + self.Ch_ratio)

        # 碳排放约束
        CE = np.sum(self.energy_efficiency[t, :] * x_econ[t, :] * (1 - np.sum(self.a, axis=0)) * self.ief[t, :]) * (
                1 + self.Cp_ratio + self.Ch_ratio)
        if t >= 10:  # 从第11年（t=10）开始逐年下降
            g[constraint_index] = (CE - previous_ce) / previous_ce  # 约束为当前年的碳排放量小于上一年的碳排放量
            constraint_index += 1

        #碳排放公平约束
        for i in range(self.M):
            if t == 0:
                previous_ce_ratio = self.initial_energy_consumption[i] * self.initial_ief[i] / previous_ce
            else:
                previous_ce_ratio = self.energy_efficiency[t - 1, i] * x_econ[t - 1, i] * (1 - np.sum(self.a[:, i])) * self.ief[t - 1, i] / previous_ce
            current_ce_ratio = self.energy_efficiency[t, i] * x_econ[t, i] * (1 - np.sum(self.a[:, i])) * self.ief[t, i] / CE
            g[constraint_index] = (1 + self.Lci) - current_ce_ratio/ previous_ce_ratio
            constraint_index += 1
            g[constraint_index] = current_ce_ratio / previous_ce_ratio - (1 + self.Uci)
            constraint_index += 1


        if t in [9, 19,29,39]:  # 在2030, 2040, 2050, 2060年进行限制
            g[constraint_index] = (CE - self.CE_lim[t // 10]) / self.CE_lim[t // 10]
            constraint_index += 1

        return constraint_index

    def add_pollutant_constraints(self, x_econ, t, g, constraint_index):
        """添加污染物排放约束"""
        # 计算排放
        PE_se = np.sum(self.se[t, :] * (x_econ[t, :]) * (1 - np.sum(self.a, axis=0))) * (1 + self.residential_se_ratio)
        PE_pe = np.sum(self.pe[t, :] * (x_econ[t, :]) * (1 - np.sum(self.a, axis=0))) * (1 + self.residential_pe_ratio)
        PE_ne = np.sum(self.ne[t, :] * (x_econ[t, :]) * (1 - np.sum(self.a, axis=0))) * (1 + self.residential_ne_ratio)

        if t == 0:
            previous_pe_se = self.initial_pe_se
            previous_pe_pe = self.initial_pe_pe
            previous_pe_ne = self.initial_pe_ne
        else:
            previous_pe_se = np.sum(self.se[t - 1, :] * (x_econ[t - 1, :]) * (1 - np.sum(self.a, axis=0))) * (
                        1 + self.residential_se_ratio)
            previous_pe_pe = np.sum(self.pe[t - 1, :] * (x_econ[t - 1, :]) * (1 - np.sum(self.a, axis=0))) * (
                        1 + self.residential_se_ratio)
            previous_pe_ne = np.sum(self.ne[t - 1, :] * (x_econ[t - 1, :]) * (1 - np.sum(self.a, axis=0))) * (
                        1 + self.residential_ne_ratio)

        # 污染物排放下降率约束
        g[constraint_index] = (PE_se - previous_pe_se) / previous_pe_se
        constraint_index += 1
        g[constraint_index] = (PE_pe - previous_pe_pe) / previous_pe_pe
        constraint_index += 1
        g[constraint_index] = (PE_ne - previous_pe_ne) / previous_pe_ne
        constraint_index += 1

        return constraint_index

    def _evaluate(self, x, out, *args, **kwargs):
        # 分解变量数组
        x_econ = x[:self.M * self.T].reshape((self.T, self.M))  # 经济产出变量

        GDP = np.zeros(self.T)
        CE = np.zeros(self.T)
        PE_se = np.zeros(self.T)
        PE_pe = np.zeros(self.T)
        PE_ne = np.zeros(self.T)
        PE = np.zeros(self.T)
        g = np.zeros(self.n_constr)  # 用于存储约束违反度的数组
        constraint_index = 0

        for t in range(self.T):
            # 经济模块约束
            constraint_index = self.add_economic_constraints(x_econ, t, g, constraint_index)
            # 能源约束
            constraint_index = self.add_energy_constraints(x_econ, t, g, constraint_index)
            # 碳排放约束
            constraint_index = self.add_carbon_constraints(x_econ, t, g, constraint_index)

            # 污染物模块约束
            constraint_index = self.add_pollutant_constraints(x_econ, t, g, constraint_index)

            # 计算GDP
            GDP[t] = np.sum(x_econ[t, :] * (1 - np.sum(self.a, axis=0)))
            # 计算碳排放

            CE[t] = np.sum(
                self.energy_efficiency[t, :] * x_econ[t, :] * (1 - np.sum(self.a, axis=0)) * self.ief[t, :]) * (
                            1 + self.Cp_ratio + self.Ch_ratio)

            # 计算污染物排放
            PE_se[t] = np.sum(self.se[t, :] * (x_econ[t, :]) * (1 - np.sum(self.a, axis=0))) * (
                        1 + self.residential_se_ratio)
            PE_pe[t] = np.sum(self.pe[t, :] * (x_econ[t, :]) * (1 - np.sum(self.a, axis=0))) * (
                        1 + self.residential_pe_ratio)
            PE_ne[t] = np.sum(self.ne[t, :] * (x_econ[t, :]) * (1 - np.sum(self.a, axis=0))) * (
                        1 + self.residential_ne_ratio)
            PE[t] = PE_se[t] + PE_pe[t] + PE_ne[t]

        # 计算目标函数值
        f1 = -np.sum(GDP)  # 负的 GDP 值
        f2 = np.sum(CE)  # 碳排放总量
        f3 = np.sum(PE)  # 污染物排放总量

        # 计算违反度总和，而不是违反的数量
        violation_degree = np.sum(g[g > 0])  # 只计算违反的约束，并累加违反的程度
        penalty = self.penalty_weight * violation_degree
        f1 += 5 * penalty
        f2 += 0.01 * penalty
        f3 += 0.05 * penalty
        out["F"] = np.array([f1, f2, f3]) # 更新目标函数值
        out["G"] = g  # 约束违反度数组


# 读取文件
rootpath = '/gpfs/share/home/gcsyhjxycsyjjdlx/BIT0812/BITtest2'
Io_paramater = pd.read_excel(os.path.join(rootpath, 'iocoeff.xlsx'), header=None)  # 投入产出系数矩阵
Eco_growthrate = pd.read_excel(os.path.join(rootpath, 'minGDPincre1.xlsx'))  # 经济增长限制，包含三种情景
Energy = pd.read_excel(os.path.join(rootpath, 'enstrucout1.xlsx'), sheet_name=0)  # 总能耗及能源结构预测能源结构预测
Energy_initial = pd.read_excel(os.path.join(rootpath, '20primaryenergy.xlsx'), sheet_name=0)  # 初始各产业各能源消耗
Mu = pd.read_excel(os.path.join(rootpath, 'enstrucout1.xlsx'), sheet_name=1, header=None)  # 能源消费总体变化率

carbon_energy_index = pd.read_excel(os.path.join(rootpath, 'encarboncoeffpre2.xlsx'))  # 碳排放系数（产业）
carbon_changerate = pd.read_excel(os.path.join(rootpath, 'maxcarbonchange.xlsx'), sheet_name=0)  # 碳排放逐年下降比率

Plt_Initial = pd.read_excel(os.path.join(rootpath, 'polludepart.xlsx'), sheet_name=0)  # 初始污染物排放
SE_rate = pd.read_excel(os.path.join(rootpath, 'polluintensitypre1.xlsx'), sheet_name=0)
NE_rate = pd.read_excel(os.path.join(rootpath, 'polluintensitypre1.xlsx'), sheet_name=1)
PE_rate = pd.read_excel(os.path.join(rootpath, 'polluintensitypre1.xlsx'), sheet_name=2)  # 污染物排放强度数据（三种）
Plt_changerate = pd.read_excel(os.path.join(rootpath, 'minpollureduc.xlsx'))  # 污染物最低下降率（三种）
energygrowth_chushi = pd.read_excel(os.path.join(rootpath, 'enstrucout1.xlsx'), sheet_name=2,
                                    header=None)  # 各类型能源的增长率初始化
Indus_Energy_rate = pd.read_excel(os.path.join(rootpath, 'enintensitypre2.xlsx'), header=None)  # 各产业能耗系数

# 各产业能耗系数
M = 16  # 部门数量
T = 40  # 年份数，从2021到2030
a = Io_paramater.to_numpy()  # 投入产出系数矩阵
alpha = Eco_growthrate.iloc[:,3].to_numpy() # 年增长率
# A = np.random.rand(M, T)  # 部门增长限制
initial_gdp = 1016422.0014 # 初始GDP
initial_structure = [82174.7242, 30057.0804, 30701.3686, 14447.6357, 9708.6186, 6406.4400, 47087.9771, 23058.0001,
                     40306.9522, 71239.6892, 9680.6870, 25841.9919, 72009.2940, 109912.3998, 45677.4937, 398111.6489]  # 初始经济结构
Ec = Energy.iloc[:,10].to_numpy()#  总能耗限制

Ues = 0.12  # 能耗上限增长率
Les = -0.12  # 能耗下限增长率
mu = Mu.to_numpy() # 各年能源变化率
initial_energy_consumption =  np.array([7284.241963, 9202.209136, 5093.968231, 4776.722596, 3317.02818, 607.2445321, 74906.85583, 37063.03948, 97992.61266,
 9423.906334, 1578.310275, 127563.336, 8080.430732, 41128.30604, 6791.0032, 16818.1964]) # 初始的各部门能源消费

Lp = -0.14  # 经济结构稳定性下限
Up = 0.14  # 经济结构稳定性上限

ief = carbon_energy_index.to_numpy()  # 碳排放系数
Cp_ratio = 0.065  # 过程排放占总碳排放的比例
Ch_ratio = 0.042  # 居民排放占总碳排放的比例
phi_c = carbon_changerate.to_numpy()  # 碳排放下降率
Lci = -0.28  # 各产业碳排放占比变化下限
Uci = 0.28  # 各产业碳排放占比变化上限
CE_lim = {0: 10800, 1: 8500, 2: 5600, 3: 1500}  # 碳排放限制
initial_ief = np.array([0.017673245, 0.020179857, 0.02036254, 0.018951525, 0.019200506, 0.016657853, 0.013167936
,0.020194604, 0.024486214,0.018769034, 0.017718598, 0.019580506,0.018411416, 0.020155891, 0.020087538, 0.019749719])

initial_pe_se = 318.22  # 2020年主要污染物排放初始值
initial_pe_pe = 613.38  # 2020年次要污染物排放初始值
initial_pe_ne = 1175.72  # 2020年其他污染物排放初始值
se = SE_rate.to_numpy()  # SO2排放系数
pe = PE_rate.to_numpy()  # PM排放系数
ne = NE_rate.to_numpy()  # Nox排放系数
phi_s = Plt_changerate.iloc[:,1].to_numpy()  # SO2排放最低下降率
phi_p = Plt_changerate.iloc[:,3].to_numpy()  # PM排放最低下降率
phi_n = Plt_changerate.iloc[:,2].to_numpy()  # Nox排放最低下降率
demand_ratio = [0.282971614, -0.518407172, 0.456856845, 0.375477907, 0.155875186, 0.334741028, 0.036359921,	0.040577818, 0.0357022,
                0.351860499, 0.071567836, 0.144624663,	0.947560612, 0.405834785, 0.178411725, 0.522634353,	0.376588192] # 最终需求的比例
residential_energy_ratio= 0.09 # 居民能耗占比
residential_se_ratio = 0.055  # SO2居民排放占比
residential_pe_ratio = 0.01 # PM居民排放占比
residential_ne_ratio = 0.084   # Nox居民排放占比
energy_efficiency = Indus_Energy_rate.to_numpy() # 各产业能耗系数


# 创建问题实例
problem = IntegratedOptimizationProblem(
    M, T, a, alpha, initial_gdp, initial_structure, Ec, Ues, Les, mu,
    initial_energy_consumption, Lp, Up, initial_ief, ief, Cp_ratio, Ch_ratio, phi_c, Lci, Uci, CE_lim,
    se, pe, ne, phi_s, phi_p, phi_n, demand_ratio, residential_energy_ratio, residential_se_ratio, residential_pe_ratio,
    residential_ne_ratio, initial_pe_se, initial_pe_pe, initial_pe_ne, energy_efficiency
)

# 自定义初始化策略
custom_sampling = CustomSampling(M, T, alpha, initial_structure, problem.xl, problem.xu)


# def calculate_diversity(pop):
#     dist_matrix = np.linalg.norm(pop - pop[:, np.newaxis], axis=2)
#     mean_dist = np.mean(dist_matrix, axis=1)
#     diversity = np.mean(1 / (mean_dist + 1e-6))  # 加入一个小的常数以避免除以零
#     return diversity

class PrintInfoCallback(Callback):
    def __init__(self, problem, max_gen):
        super().__init__()
        self.problem = problem
        self.max_gen = max_gen

    def notify(self, algorithm):
        # 检查并将种群中所有个体的数据类型转换为 float64
        for ind in algorithm.pop:
            ind.X = ind.X.astype(np.float64)

        if algorithm.n_gen == 1:
            # 打印初始种群的个体
            print("Initial Population:")
            for i, ind in enumerate(algorithm.pop):
                X = ind.X
                # 为F和G创建字典引用
                out = {"F": np.zeros(self.problem.n_obj), "G": np.zeros(self.problem.n_constr)}
                # 调用问题类的_evaluate方法来评估个体
                self.problem._evaluate(X, out=out)
                # 从out字典获取F和G的值
                F = out["F"]
                G = out["G"]
                print(f"Individual {i}: X = {X}, F = {F}, G = {G}")
        # else:
        #     # 计算种群多样性
        #     diversity = calculate_diversity(np.array([ind.X for ind in algorithm.pop]))
        #     print(f"Generation {algorithm.n_gen}: Diversity = {diversity}")

        # 打印当前种群的个体
        # 动态调整罚函数权重
        self.problem.penalty_weight = (500 + 100 * (algorithm.n_gen**2) / self.max_gen)

        # 打印个体的信息
        cv_values = np.array([ind.CV for ind in algorithm.pop])
        min_cv_index = np.argmin(cv_values)
        best_individual = algorithm.pop[min_cv_index]
        violated_constraints_indices = np.where(best_individual.G > 0)[0]
        violated_constraints_values = best_individual.G[violated_constraints_indices]
        print(f"Generation {algorithm.n_gen}: Minimum Constraint Violations: {best_individual.CV}")
        print(
            f"Generation {algorithm.n_gen}: Individual {min_cv_index}: F = {best_individual.F}, G = {best_individual.G}")
        print(f"Generation {algorithm.n_gen}: Number of violated constraints: {len(violated_constraints_indices)}")
        # print(f"Generation {algorithm.n_gen}: Individual {min_cv_index} Genes: {best_individual.X}")  # 新增打印个体基因值的行
        for idx, value in zip(violated_constraints_indices, violated_constraints_values):
            print(f"  Index {idx}: Violation {value}")
        print(f"Generation {algorithm.n_gen}: Individual {min_cv_index} Genes: {best_individual.X}")  # 打印个体基因值


# 使用默认的 NSGA-II 算法
max_gen = 45000  # 设置最大代数
algorithm = NSGA2(pop_size=1600,
                  n_offsprings=2200,
                  sampling=custom_sampling,
                  crossover=SBX(prob=0.8, eta=15),
                  mutation=PM(prob=0.2, eta=20),
                  eliminate_duplicates=True)

# 添加回调函数
callback = PrintInfoCallback(problem, max_gen=max_gen)


# 定义并行评估函数
def evaluate_individual(problem, x):
    out = {"F": None, "G": None}
    problem._evaluate(x, out=out)
    return out


# 定义自定义的并行评估器类
class ParallelEvaluator:
    def __init__(self, n_jobs=8):
        self.n_jobs = n_jobs
        self.n_eval = 0  # 初始化评估次数为0

    def eval(self, problem, pop, **kwargs):
        results = Parallel(n_jobs=self.n_jobs)(delayed(evaluate_individual)(problem, ind.X) for ind in pop)
        for ind, res in zip(pop, results):
            ind.F = res["F"]
            ind.G = res["G"]
        # 更新评估次数
        self.n_eval += len(pop)


# 创建问题实例算法实例
evaluator = ParallelEvaluator(n_jobs=8)

# 进行优化
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', max_gen),
    save_history=False,
    verbose=True,
    eliminate_duplicates=True,
    callback=callback,  # 使用回调函数
    evaluator=evaluator)  # 自定义评估器

# 提取帕累托前沿面的解
pareto_front = res.F

# 归一化处理后的指标调整
pareto_front_norm = np.zeros_like(pareto_front)

# 正向指标 (经济产出)
pareto_front_norm[:, 0] = (pareto_front[:, 0] - pareto_front[:, 0].min()) / (
            pareto_front[:, 0].max() - pareto_front[:, 0].min())

# 负向指标 (碳排放和污染物排放)
pareto_front_norm[:, 1] = (pareto_front[:, 1].max() - pareto_front[:, 1]) / (
            pareto_front[:, 1].max() - pareto_front[:, 1].min())
pareto_front_norm[:, 2] = (pareto_front[:, 2].max() - pareto_front[:, 2]) / (
            pareto_front[:, 2].max() - pareto_front[:, 2].min())

# 设置目标函数的权重
weights = np.array([1 / 3, 1 / 3, 1 / 3])  # 示例权重，可以根据具体需求调整

# 计算加权归一化矩阵
weighted_pareto_front_norm = pareto_front_norm * weights

# 确定理想解和负理想解
ideal_solution = np.min(weighted_pareto_front_norm, axis=0)
negative_ideal_solution = np.max(weighted_pareto_front_norm, axis=0)

# 计算到理想解和负理想解的距离
distance_to_ideal = np.sqrt(np.sum((weighted_pareto_front_norm - ideal_solution) ** 2, axis=1))
distance_to_negative_ideal = np.sqrt(np.sum((weighted_pareto_front_norm - negative_ideal_solution) ** 2, axis=1))

# 计算相对接近度
relative_closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

# 找到相对接近度最大的解
best_solution_index = np.argmax(relative_closeness)

# 提取最佳解对应的决策变量
best_solution = res.X[best_solution_index]

# 提取并重塑最佳解的变量
x_econ_opt = best_solution[:M * T].reshape((T, M))  # 经济产出变量

# 输出最佳解
print("最佳解:")
print("经济产出变量:", x_econ_opt)
print("目标函数值:", pareto_front[best_solution_index])

# 保存帕累托前沿面的解
np.savetxt("/gpfs/share/home/gcsyhjxycsyjjdlx/BIT0812/BITtest2/pareto_front2060S31.csv", pareto_front, delimiter=",")
# 保存最优产出
np.savetxt("/gpfs/share/home/gcsyhjxycsyjjdlx/BIT0812/BITtest2/eco_output2060S31.csv", x_econ_opt, delimiter=",")