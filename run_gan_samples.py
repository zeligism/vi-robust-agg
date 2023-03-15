
import os
import argparse
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from codes.tasks.gan import ResNetGAN, gan_dataset
from codes.tasks.inception.inception_score import inception_score
from utils import DATA_DIR


def get_args(namespace=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("-m", "--model-path", type=str, help="Model directory.")
    parser.add_argument("-n", "--num-samples", type=int, default=10000, help="Number of samples to generate.")
    parser.add_argument("-BS", "--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--inception-score", action="store_true", default=False, help="Compute Inception Score (IS).")
    args = parser.parse_args(namespace=namespace)
    return args


class GAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, model_init=lambda: ResNetGAN(), model_path="model.pl",
                 num_samples=10000, batch_size=32, device="cpu"):
        self.device = device
        self.model_path = model_path
        self.model = model_init().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))

        # Get fake and real data and save them (for FID calculation)
        self.fake_data = self.generate_samples(num_samples, batch_size)
        self.real_data = gan_dataset(data_dir=DATA_DIR, train=True,
                                     download=True, batch_size=1)
        self.save_data(self.fake_data, dirname="fake_data")
        # self.save_data(self.real_data, dirname="real_data")

    def generate_samples(self, num_samples, batch_size):
        data = []
        print(f"Generating {num_samples} samples.")
        self.model.eval()
        for _ in tqdm(range(num_samples // batch_size + 1)):
            with torch.no_grad():
                latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
                fake = self.model.G(latent)
                data.append(fake)
        return torch.cat(data)

    def save_data(self, data, dirname="data"):
        out_dir = os.path.join(os.path.dirname(self.model_path), dirname)
        if os.path.exists(out_dir):
            print("Data already exists at:", out_dir)
        else:
            os.makedirs(out_dir)
            print("Saving data to:", out_dir)
            for i, data_i in tqdm(enumerate(data)):
                if len(data_i) == 2:
                    data_i = data_i[0]  # get img only
                save_image(data_i, f"{out_dir}/{i:04}.png", normalize=True, range=(-1,1))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def main(args):
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    gan_dataset = GAN_Dataset(model_path=args.model_path, num_samples=args.num_samples,
                              batch_size=args.batch_size, device=device)
    if args.inception_score:
        score = inception_score(gan_dataset, cuda=args.use_cuda,
                                batch_size=args.batch_size, resize=True, splits=10)
        print(f"Inception score = {score}")


if __name__ == '__main__':
    args = get_args()
    main(args)
