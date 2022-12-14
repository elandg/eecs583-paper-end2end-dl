import matplotlib.pyplot as plt

data_folder = "organized_results"
figures_folder = "figures"

amd_aux_100_epochs = [
    "cnn-1-original-tokens-100-amd.txt",
    "cnn-2-original-tokens-100-amd.txt",
    "cnn-3-original-tokens-100-amd.txt",
    "lstm-original-tokens-100-amd.txt",
]
amd_no_aux_100_epochs = [
    "no-aux-cnn-1-original-tokens-100-amd.txt",
    "no-aux-cnn-2-original-tokens-100-amd.txt",
    "no-aux-cnn-3-original-tokens-amd.txt",
    "no-aux-lstm-original-tokens-100-amd.txt",
]

nvidia_cnn_2_aux_diff_token = [
    "cnn-2-original-tokens-nvidia.txt",
    "cnn-2-ralph-tokens-nvidia.txt",
    "cnn-2-alex-tokens-nvidia.txt",
    "cnn-2-c99-tokens-nvidia.txt",
    "cnn-2-all-tokens-nvidia.txt",
]
nvidia_cnn_2_no_aux_diff_token = [
    "no-aux-cnn-2-original-tokens-nvidia.txt",
    "no-aux-cnn-2-ralph-tokens-nvidia.txt",
    "no-aux-cnn-2-alex-tokens-nvidia.txt",
    "no-aux-cnn-2-c99-tokens-nvidia.txt",
    "no-aux-cnn-2-all-tokens-nvidia.txt",
]

diff_tokenization_names = [
    "OpenCL",
    "OpenCL - C99",
    "OpenCL - C99 - GLSL - Solidity",
    "C99",
    "OpenCL + C99 + GLSL + Solidity",
]

diff_arch_names = [
    "1 Layer CNN",
    "2 Layer CNN",
    "3 Layer CNN",
    "LSTM",
]

split_b_vs_tl_amd_names = [
    "AMD Radeon HD",
    "AMD Tahiti",
    "AMD Radeon HD (TL)",
    "AMD Tahiti (TL)",
]

split_b_vs_tl_nvidia_names = [
    "NVIDIA GTX",
    "NVIDIA Tesla",
    "NVIDIA GTX (TL)",
    "NVIDIA Tesla (TL)",
]

cnn_2_b_vs_tl_amd = [
    "cnn-2-ralph-tokens-b-hd.txt",
    "cnn-2-ralph-tokens-b-tahiti.txt",
    "cnn-2-ralph-tokens-b-tl-hd.txt",
    "cnn-2-ralph-tokens-b-tl-tahiti.txt",
]

cnn_2_b_vs_tl_nvidia = [
    "cnn-2-ralph-tokens-b-gtx.txt",
    "cnn-2-ralph-tokens-b-tesla.txt",
    "cnn-2-ralph-tokens-b-tl-gtx.txt",
    "cnn-2-ralph-tokens-b-tl-tesla.txt",
]


def get_filepaths_per_tokenization(tokenization_method, platform, is_aux):
    prefix = "" if is_aux else "no-aux-"
    return [
        f"{prefix}cnn-1-{tokenization_method}-tokens-{platform}.txt",
        f"{prefix}cnn-2-{tokenization_method}-tokens-{platform}.txt",
        f"{prefix}cnn-3-{tokenization_method}-tokens-{platform}.txt",
        f"{prefix}lstm-{tokenization_method}-tokens-{platform}.txt",
    ]


def get_title(tokenization_method, platform, is_aux):
    platform = "AMD" if platform == "amd" else "NVIDIA"
    tokenization_method = {
        "original": "OpenCL",
        "ralph": "OpenCL - C99",
        "alex": "OpenCL - C99 - GLSL - Solidity",
        "c99": "C99",
        "all": "OpenCL + C99 + GLSL + Solidity",
    }[tokenization_method]
    is_aux = "(aux)" if is_aux else "(no aux)"
    return f"{platform} {tokenization_method} Tokenization {is_aux}"


def get_loss_and_accs(filename, is_aux):
    losses = []
    accs = []
    with open(filename) as fhandle:
        for line in fhandle.readlines():
            line = line.strip()
            if line.startswith("Epoch"):
                continue
            pieces = line.split()
            if len(pieces) < 6:
                continue
            loss = float(pieces[7 if not is_aux else 10])
            acc = float(pieces[10 if not is_aux else 16])
            losses.append(loss)
            accs.append(acc)
    return losses, accs


def plot_per_token(files, is_aux, title):
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(f"{title} Accuracy")
    axs[1].set_title("Loss")
    for i in range(4):
        losses, accs = get_loss_and_accs(f"{data_folder}/{files[i]}", is_aux)
        axs[0].plot(accs, label=diff_arch_names[i])
        axs[1].plot(losses, label=diff_arch_names[i])
    axs[0].legend()
    axs[1].legend()
    plt.savefig(f"{figures_folder}/{title}.png")
    plt.close()


def plot_per_token_both_aux(files1, files2, title1, title2, labels=diff_arch_names):
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(f"{title1} Accuracy")
    axs[1].set_title(f"{title2} Accuracy")
    # First plot aux
    for i in range(len(files1)):
        losses, accs = get_loss_and_accs(f"{data_folder}/{files1[i]}", True)
        axs[0].plot(accs, label=labels[i])
    # Then no aux
    for i in range(len(files2)):
        _, accs = get_loss_and_accs(f"{data_folder}/{files2[i]}", False)
        axs[1].plot(accs, label=labels[i])
    axs[0].legend()
    axs[1].legend()
    plt.savefig(f"{figures_folder}/{title1} combined.png")
    plt.close()


def plot_per_token_diff_labels(files1, files2, title1, title2, labels1, labels2, is_aux=False):
    plt.rcParams["figure.figsize"] = (8, 8)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(f"{title1} Accuracy")
    axs[1].set_title(f"{title2} Accuracy")
    # First plot aux
    for i in range(len(files1)):
        losses, accs = get_loss_and_accs(f"{data_folder}/{files1[i]}", is_aux)
        axs[0].plot(accs, label=labels1[i])
    # Then no aux
    for i in range(len(files2)):
        _, accs = get_loss_and_accs(f"{data_folder}/{files2[i]}", is_aux)
        axs[1].plot(accs, label=labels2[i])
    axs[0].legend()
    axs[1].legend()
    plt.savefig(f"{figures_folder}/{title1} combined.png")
    plt.close()


def print_acc_and_loss():
    for tokenization_method in ["original", "ralph", "alex", "all", "c99"]:
        for platform in ["amd", "nvidia"]:
            for is_aux in [True, False]:
                plot_per_token(
                    get_filepaths_per_tokenization(
                        tokenization_method,
                        platform,
                        is_aux
                    ),
                    is_aux,
                    get_title(tokenization_method, platform, is_aux)
                )
    plot_per_token(amd_aux_100_epochs, True,
                   "AMD Original Tokenization (aux) 100 Epochs")


def print_acc_aux_vs_no_aux():
    for tokenization_method in ["original", "ralph", "alex", "all", "c99"]:
        for platform in ["amd", "nvidia"]:
            plot_per_token_both_aux(
                get_filepaths_per_tokenization(
                    tokenization_method,
                    platform,
                    True
                ),
                get_filepaths_per_tokenization(
                    tokenization_method,
                    platform,
                    False
                ),
                get_title(tokenization_method, platform, True),
                get_title(tokenization_method, platform, False),
            )
    plot_per_token_both_aux(amd_aux_100_epochs,             amd_no_aux_100_epochs,
                            "AMD Original Tokenization AMD (aux) 100 Epochs", "AMD Original Tokenization AMD (no aux) 100 Epochs")
    plot_per_token_both_aux(nvidia_cnn_2_aux_diff_token, nvidia_cnn_2_no_aux_diff_token, "2 Layer CNN Tokenization Methods NVIDIA (aux)",
                            "2 Layer CNN Tokenization Methods NVIDIA (no aux)", labels=diff_tokenization_names)
    plot_per_token_diff_labels(cnn_2_b_vs_tl_amd, cnn_2_b_vs_tl_nvidia, "Part B vs Transfer Learning AMD",
                               "Part B vs Transfer Learning NVIDIA", split_b_vs_tl_amd_names, split_b_vs_tl_nvidia_names)


def main():
    print_acc_aux_vs_no_aux()
    # print_acc_and_loss()


if __name__ == "__main__":
    main()
