import matplotlib.pyplot as plt
import numpy as np


# 在每个柱子上显示数值
def add_value_labels(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

def serving():
    labels = ['total-time[s]', 'throughtput[req/s]', 'first-token-latency[ms]', 'token-generate-latency[ms/token]']
    vllm = [90.69, 1.54, 32.92, 26.64]  # 
    lite_infer = [69.59, 1.84, 25.21, 24.74]  # 

    # 设置 x 轴位置
    x = np.arange(len(labels))  # [0, 1, 2, 3]
    width = 0.35  # 每个柱子的宽度

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 画柱状图
    bars1 = ax.bar(x - width/2, vllm, width, label='vllm', color='#66b3ff')
    bars2 = ax.bar(x + width/2, lite_infer, width, label='lite_infer', color='#ff9999')

    # 添加标题和标签
    ax.set_xlabel('')
    ax.set_ylabel('performance')
    ax.set_title('vllm and lite_infer performance comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    add_value_labels(bars1, ax)
    add_value_labels(bars2, ax)

    # 美化图形
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)

    # 显示图形
    plt.tight_layout()
    plt.show()
    plt.savefig("performance.png")

def lite_infer_vllm():
    labels = ['time[s]', 'throughtput[req/s]', 'output-token[x10 token/s]','first-token-latency[ms]', 'token-generate-latency[ms/token]', 'prefix-caching-hit-rate[%]']
    standard = np.array([100.98, 1.26, 66.1, 34.13, 34.38, 0])  # 
    w_prefix_caching = np.array([95.86, 1.33, 69.6, 32.54, 33.12, 11.39])  # 
    w_prefix_caching_CUDA_graph = np.array([69.59, 1.84, 95.9, 25.21, 24.74, 11.39])
    vllm = np.array([83.08, 1.54, 80.42, 32.92, 26.64, 26.6])

    # 设置 x 轴位置
    x = 1.6*np.arange(len(labels))  # [0, 1, 2, 3]
    width = 0.28  # 每个柱子的宽度

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(16, 9))

    # 画柱状图
    bars1 = ax.bar(x - 2*width, standard, width, label='lite-infer-standard', color='#66b3ff')
    bars2 = ax.bar(x - 1*width, w_prefix_caching, width, label='lite-infer-prefix caching', color='#ff9999')
    bars3 = ax.bar(x, w_prefix_caching_CUDA_graph, width, label='lite-infer-prefix caching+cuda graph', color='#88e399')
    bars4 = ax.bar(x + 1*width, vllm, width, label='vllm', color='#FFF899')

    # 添加标题和标签
    ax.set_xlabel('')
    ax.set_ylabel('performance')
    ax.set_title('lite_infer performance compare')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    add_value_labels(bars1, ax)
    add_value_labels(bars2, ax)
    add_value_labels(bars3, ax)
    add_value_labels(bars4, ax)

    # 美化图形
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 105)

    # 显示图形
    plt.tight_layout()
    plt.savefig("lite_infer_performance_compare.png", dpi=400)
    #----------------------------------------------------------------------------------------

    # normalization
    labels = ['time', 'throughtput', 'output-token','first-token-latency', 'token-generate-latency', 'prefix-caching-hit-rate']
    vllm_ = np.array([100, 100, 100, 100, 100, vllm[-1]])  #
    standard[:-1] = standard[:-1] / vllm[:-1] * 100
    w_prefix_caching[:-1] = w_prefix_caching[:-1] / vllm[:-1] * 100
    w_prefix_caching_CUDA_graph[:-1] = w_prefix_caching_CUDA_graph[:-1]/ vllm[:-1] * 100
    # 设置 x 轴位置
    x = 1.6*np.arange(len(labels))  # [0, 1, 2, 3]
    width = 0.28  # 每个柱子的宽度

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(16, 9))

    # 画柱状图
    bars1 = ax.bar(x - 2*width, standard, width, label='lite-infer-standard', color='#66b3ff')
    bars2 = ax.bar(x - 1*width, w_prefix_caching, width, label='lite-infer-prefix caching', color='#ff9999')
    bars3 = ax.bar(x, w_prefix_caching_CUDA_graph, width, label='lite-infer-prefix caching+cuda graph', color='#88e399')
    bars4 = ax.bar(x + 1*width, vllm_, width, label='vllm', color='#FFF899')

    # 添加标题和标签
    ax.set_xlabel('')
    ax.set_ylabel('performance[%]')
    ax.set_title('lite_infer performance compare')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    add_value_labels(bars1, ax)
    add_value_labels(bars2, ax)
    add_value_labels(bars3, ax)
    add_value_labels(bars4, ax)

    # 美化图形
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 140)

    # 显示图形
    plt.tight_layout()
    plt.savefig("lite_infer_performance_compare_2.png", dpi=400)




if __name__ == "__main__":

    lite_infer_vllm()