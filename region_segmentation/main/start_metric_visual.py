from process import Visualizer


def main():
    # 工作目录
    # root = '~/output/diff_merge'
    # names = [f'auto-save-epoch-{i}' for i in [11, 17]] + ['v5-3-15', 'v5-3-19', 'v5-2-17']
    root = '~/output/latest'
    names = [f'auto-save-epoch-{i}' for i in [29, 40]]
    # names = [f'auto-save-epoch-{i + 1}' for i in range(20)]
    # 快速操作，选择性的展示评估结果
    V = Visualizer(root=root)
    # V.diff(names=names)
    # V.dice(names=names)
    V.best()


if __name__ == '__main__':
    main()
