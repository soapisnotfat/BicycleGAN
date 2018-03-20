import argparse
from solver import Solver


def main(opts):
    solver = Solver(root=opts.root,
                    result_dir=opts.result_dir,
                    weight_dir=opts.weight_dir,
                    load_weight=opts.load_weight,
                    batch_size=opts.batch_size,
                    test_size=opts.test_size,
                    test_img_num=opts.test_img_num,
                    img_size=opts.img_size,
                    num_epoch=opts.num_epoch,
                    save_every=opts.save_every,
                    lr=opts.lr,
                    beta_1=opts.beta_1,
                    beta_2=opts.beta_2,
                    lambda_kl=opts.lambda_kl,
                    lambda_img=opts.lambda_img,
                    lambda_z=opts.lambda_z,
                    z_dim=opts.z_dim)

    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='dataset/edges2shoes', help='Data location')
    parser.add_argument('--result_dir', type=str, default='result_img', help='Result images location for intermediate check')
    parser.add_argument('--weight_dir', type=str, default='weight', help='Weight location')
    parser.add_argument('--batch_size', type=int, default=75, help='Training batch size')
    parser.add_argument('--test_size', type=int, default=20, help='Test batch size for intermediate check')
    parser.add_argument('--test_img_num', type=int, default=5, help='How many images do you want to generate for intermediate check?')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta2 for Adam')
    parser.add_argument('--lambda_kl', type=float, default=0.01, help='Lambda for KL Divergence')
    parser.add_argument('--lambda_img', type=float, default=10, help='Lambda for image reconstruction')
    parser.add_argument('--lambda_z', type=float, default=0.5, help='Lambda for z reconstruction')
    parser.add_argument('--z_dim', type=int, default=8, help='Dimension of z')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epoch')
    parser.add_argument('--save_every', type=int, default=300, help='How often do you want to see the intermediate result?')
    parser.add_argument('--load_weight', action='store_true', help='Load weight or not')

    args = parser.parse_args()
    main(args)
