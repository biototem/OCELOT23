from process import do_calculate, do_evaluate
from utils import Timer


def main():
    # 用于评估的模型名称
    # model_names = ['score', 'loss']
    model_names = [f'auto-save-epoch-{epoch+1}' for epoch in range(0, 20)]

    # # 工作目录
    # root = '~/output/v4-3'
    for root in [
        '~/output/v1-2-s',
        # '~/output/diff_merge',
        # '/media/totem_disk/totem/jizheng/breast_competation/output/experiment/eff-simple-aux',
        # '/media/totem_disk/totem/jizheng/breast_competation/output/experiment/eff-norm-na',
        # '/media/totem_disk/totem/jizheng/breast_competation/output/experiment/van-norm-aux',
    ]:
        # 耗时操作，使用模型预测数据集，生成并保存原始统计数据（模型 X 在图片 Z 中 gt=A 而 pr=B 的像素数量）
        # 一并保存计算时的训练集、验证集分配情况
        with Timer() as T:
            do_calculate(model_names=model_names, root=root, T=T, visual=False)

        # 快速操作，根据已生成的原始统计数据，继续生成评估表
        do_evaluate(root=root)


if __name__ == '__main__':
    main()
