import matplotlib.pyplot as plt


def plot_latency(results, output_path):
    keys = [f"{r['model']}\n{r['device']}" for r in results]
    values = [r['decode_latency_s'] for r in results]
    plt.figure(figsize=(8, 5))
    plt.bar(keys, values, color='skyblue')
    plt.ylabel('Decode Latency (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)


def plot_throughput(results, output_path):
    keys = [f"{r['model']}\n{r['device']}" for r in results]
    values = [r['throughput_tokens_s'] for r in results]
    plt.figure(figsize=(8, 5))
    plt.bar(keys, values, color='salmon')
    plt.ylabel('Throughput (tokens/s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
