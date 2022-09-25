# 定义搜索参数，采样类型，和采样的范围
search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}
"""采样类型"""
# choice：在列表里随机选取
# randint：在规定上下限中随机选取整数
# uniform: 在规定上下限中均匀采样
# quniform：使用函数 clip(round(uniform(low, high) / q) * q, low, high)进行参数选取，即在上下限中按q的间距选取参数
# loguniform：对数均匀分布 exp(uniform(log(low), log(high)))，该变量优化时约束为正。
# qloguniform: 变量值使用clip(round(loguniform(low, high) / q) * q, low, high) 确定，其中clip 操作用于将生成的值约束在范围内
# normal: 按正态分布采样
# qnormal：变量值使用round(normal(mu,sigma)/q)*q确定
# lognormal： 根据exp(normal(mu,sigma))确定变量值，呈现正态分布
# qlognormal： 根据round(exp(normal(mu, sigma))/q)*q确定

# 导入实验容器
from nni.experiment import Experiment
# 本地执行
experiment = Experiment('local')

# 配置执行文件
experiment.config.trial_command = 'python model.py'
# 配置执行文件夹
experiment.config.trial_code_directory = '.'
# 配置搜索空间
experiment.config.search_space = search_space
# 配置优化器，TPE
experiment.config.tuner.name = 'TPE'
"""优化器类型"""
# TPE：Tree-structured Parzen Estimator，一种典型的贝叶斯优化算法，是一个轻量级的优化器，没有额外的依赖，支持所有的搜索空间。
#      缺点是TPE无法发现不同超参数之间的关系
# Random：随机搜索，支持所有的搜索空间类型
# Grid Search：网格搜索法，将搜索空间划分为均匀间隔的网格，并执行蛮力遍历，支持所有的搜索空间类型。
#              推荐在搜索空间小的时候，以及当你想找到严格最优的超参数时。
# Anneal：退火算法，简单的退火算法是从先验采样开始，但随着时间的推移，趋向于越来越接近观察到的最佳点的进行采样。
#         该算法是随机搜索的一种简单变体，利用响应面的平滑度。退货速率不是自适应的，需要手动配置
# Evolution：朴素进化算法。来自于图像分类器的大规模进化。基于搜索空间随机初始化一个种群。
#             对于每一代，他选择更好的并对其进行一定的突变以获取下一代，需要多次试验，但扩展新功能非常简单易行。
# SMAC：基于序列模型的优化（SMBO）。改编了以前最突出的模型类（高斯随机进程模型），并将随机森林的模型类引入SMBO，已处理分类参数。
#      注意，SMAC需要通过pip install nni[SMAC]安装
# Batch：批处理优化器允许用户为他们的试用代码简单的提供几种配置（即超参数的选择）。完成所有配置侯，实验完成了。
#       Batch 仅支持空间规范中的类型选择
# Hyperband：尝试使用有限的资源来探索尽可能多的配置，并返回最有希望的配置作为最终结果。基本思想是生成许多配置并运行他们进行少量试验。
#            一半最不可能的配置被丢弃，其余配置与选择的新配置一起进一步训练。这些配置的规模对资源限制（如分配的搜索时间）很敏感。
# Metis：优势：虽然大多数工具仅预测最佳配置，但Metris为您提供两个输出：（a）最佳配置的当前预测，以及（b）对下一次实验的建议。
#       虽然大多数工具假设训练数据集没有噪声数据，但Metis实际上会告诉你是否需要重新采样待定的超参数。
# BOHB：是Hyperband的后续工作。它针对Hyperband的弱点，即新配置是随机生成的，无需利用已完成的试验。
#       BOHB通过构建多个TPE模型来利用已完成的试验，通过这些模型生成一定比例的新配置。
# GP：Gaussian Process优化器是基于序列模型的优化方法，以GP为代理
# PBT：是一种简单的异步优化算法，有效的利用固定的计算预算来联合优化模型群及其超参数以最大化性能
# DNGO：使用神经网络作为GP的替代方法来模拟贝叶斯优化中的函数的分布。


# 使用评估器，提高计算资源的效率，但可能会略微降低优化器的预测准确性。
# 评估器类型
# Medianstop：如果超参数集在任何步骤中的表现都比中值差，则停止。
# Curve Fitting： 如果学习曲线可能会收敛到次优结果，则停止。
experiment.config.assessor.name = 'Medianstop'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
# 定义最大优化次数和每次尝试的优化目标数
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
# 定义最大优化时间
# experiment.config.max_experiment_duration = "1h"


# 开始自优化，并在http://127.0.0.1:8080展示
experiment.run(8080)

# input('Press enter to quit')
experiment.stop()