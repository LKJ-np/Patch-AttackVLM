import argparse
import random
import clip
import torchvision
import matplotlib.pyplot as plt
from load_data import *
from patch_config import *


DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

def generate_patch(self, type):
    """
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """
    if type == 'gray':
        adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

    return adv_patch_cpu

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)

def _show_images(inps1):

    image2 = inps1[0].permute([1,2,0]) # c,h,w to h,w,c

    image2[image2<=0] = 0
    image2[image2>=1] = 1
    image2 = Image.fromarray((image2.detach().cpu().numpy()*255).astype(np.uint8))
    plt.imshow(image2)
    plt.show()

if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/16", type=str)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--output", default="../Adam_1000_64x64_0.35_ViT-B16/", type=str,help='')
    parser.add_argument("--cle_data_path", default='../data/out_cle_1000/',type=str, help='')
    parser.add_argument("--tgt_data_path",default='../data/out_tag_1000/',type=str, help='')
    args = parser.parse_args()
    args.patch_applier = PatchApplier().cuda()

    # load clip_model params
    alpha = args.alpha
    epsilon = args.epsilon
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)

    # ------------- pre-processing cle_images/text ------------- #
    # preprocess cle_images
    # 创建一个图像预处理的管道，该管道包括多个处理步骤，用于将输入图像转换为模型所需的格式。
    # 具体的处理步骤包括图像的大小调整、中心裁剪、转换为 RGB 格式以及转换为张量形式。
    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.input_res,
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(args.input_res),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img))
        ]
    )
    cle_data_path = args.cle_data_path
    clean_data = ImageFolderWithPaths(cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
    data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=12, drop_last=False)
    data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=12, drop_last=False)
    # 创建一个图像预处理管道，用于将图像转换为 CLIP 模型所需的输入格式。
    # 具体处理步骤包括将图像大小调整为 CLIP 模型的输入分辨率、将像素值归一化到 [0, 1] 范围内、中心裁剪图像、
    # 以及对图像进行归一化处理（使用 CLIP 模型预定义的均值和标准差）。
    # CLIP imgs mean and std.代理模型
    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(clip_model.visual.input_resolution,
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                          antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # CLIP imgs mean and std.
        ]
    )
    inverse_normalize = torchvision.transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

    # start attack
    # 遍历data_loader_imagenet（cle图片）和data_loader_target（目标图片）中的图像数据。
    # image_org是原始图像，image_tgt是目标图像。
    # path包含了图像的路径信息。
    # args.batch_size是每个批次的图像数量，args.num_samples是要处理的总图像数量。
    # 在这个循环中，遍历干净数据集和目标数据集的图像对，在每次迭代中，计算图像之间的相似度，并通过梯度反向传播来生成对抗样本。
    for i, ((image_org, _, path), (image_tgt, _, _)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
        if args.batch_size * (i + 1) > args.num_samples:
            break

        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)
        # get tgt featutres
        with torch.no_grad():
            # 首先通过调用clip_preprocess函数对目标图像（image_tgt）进行预处理。
            # 预处理可能包括调整图像大小、归一化或标准化等操作，以确保与 CLIP 模型的输入要求相符。
            # 然后，使用clip_model.encode_image函数对预处理后的目标图像进行编码，以获取图像的特征向量。这个特征向量可以看作是对图像的抽象表示，反映了图像在共享嵌入空间中的位置和特征。
            cle_image_features = clip_model.encode_image(clip_preprocess(image_org))
            # 最后，对获得的目标图像特征向量进行归一化处理，通过除以其自身的L2 范数（欧氏距离）来得到单位长度的向量。
            # 这个步骤可以提高特征向量的可比性和稳定性，使得它们更适合用于计算相似性或进行其他相关任务。
            cle_image_features = cle_image_features / cle_image_features.norm(dim=1, keepdim=True)

        # -------- get adv image -------- #
        # 这段代码的目标是通过迭代优化对抗扰动，使得对抗图像与目标图像在CLIP模型中的特征表示之间的相似度最大化。这样做可以使对抗图像更接近于目标图像，从而欺骗CLIP模型的分类或相似度评估。
        # 首先，创建一个与原始图像（image_org）形状相同的全零张量delta，并设置requires_grad=True以启用梯度计算。delta表示对原始图像的扰动，我们将在每个迭代步骤中对其进行调整以实现攻击目标。
        # delta = torch.zeros_like(image_org, requires_grad=True)
        # batch_size为1，16行16列的4维张量
        # 生成1套3层n行n列的4维张量。3为chanel
        adv_patch_cpu = torch.rand((1, 3, 64, 64))
        adv_patch_cpu = adv_patch_cpu.to(device)
        adv_patch_cpu = torch.rand_like(adv_patch_cpu, requires_grad=True)

        # 动量
        momentum = torch.zeros_like(adv_patch_cpu).cuda()

        # Adam优化器
        optimizer = optim.Adam([adv_patch_cpu], lr=0.35, amsgrad=True)

        # 归一化
        # 图像归一化，将图像的各像素值归一化到0~1区间。
        image_org /= 255
        args.nps_calculator = NPSCalculator("../non_printability/30values.txt",64).cuda()
        args.total_variation = TotalVariation().cuda()
        for j in range(args.steps):
            nps = args.nps_calculator(adv_patch_cpu)
            tv = args.total_variation(adv_patch_cpu)

            nps_loss = nps * 0.01
            tv_loss = tv*2.5

            loss =nps_loss
            loss =nps_loss + tv_loss

            img = image_org.clone()

            adv_image = args.patch_applier(img, adv_patch_cpu, do_rotate=True, rand_loc=True)

            # 然后，通过调用clip_model.encode_image函数对预处理后的对抗图像进行编码，
            # 得到对抗图像的特征向量adv_image_features。与之前的目标图像特征向量一样，对对抗图像特征向量也进行了归一化处理。
            adv_image_features = clip_model.encode_image(adv_image)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

            # 接下来，通过计算对抗图像特征向量与目标图像特征向量之间的余弦相似度，
            # 使用torch.sum和torch.mean函数来计算。这个相似度被用作损失函数的一部分，以衡量对抗图像与目标图像的相似程度。
            embedding_sim =torch.mean(torch.sum(adv_image_features * cle_image_features,
                                                 dim=1))  # computed from normalized features (therefore it is cos sim.)
            # 然后，使用反向传播（embedding_sim.backward()）计算关于delta的梯度，将梯度值存储在grad中，并且根据梯度信息和一些超参数进行对delta的更新。
            # 具体而言，使用torch.sign函数计算grad的符号，乘以alpha并与delta相加得到新的d值。d还通过torch.clamp函数进行限制，确保其取值在-epsilon和epsilon之间。

            # embedding_sim.backward(retain_graph=True)
            # embedding_sim.backward(retain_graph=True)


            loss += embedding_sim
            loss.backward(torch.ones_like(loss))

            optimizer.step()
            adv_patch_cpu.grad.zero_()
            adv_patch_cpu.data.clamp_(0, 1)

            #grad = adv_patch_cpu.grad.detach()
            #d = torch.clamp(adv_patch_cpu + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            #grad += momentum
            #momentum = grad
            # 更新后的d值赋给delta.data，并将delta.grad置零以准备下一次迭代。在每次迭代结束后，
            # 打印出当前迭代步骤的一些统计信息，如嵌入相似度（embedding similarity）以及对扰动d的最大值和平均值的绝对值。
            #adv_patch_cpu.data = d
            #adv_patch_cpu.grad.zero_()

            print(
                f"iter {i}/{args.num_samples // args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(adv_patch_cpu.data)).item():.3f}, mean delta={torch.mean(torch.abs(adv_patch_cpu.data)).item():.3f}")
            # print(
            #     f"iter {i}/{args.num_samples // args.batch_size} step:{j:3d}, embedding similarity={loss.item():.5f}, max delta={torch.max(torch.abs(adv_patch_cpu.data)).item():.3f}, mean delta={torch.mean(torch.abs(adv_patch_cpu.data)).item():.3f}")

        # save imgs
        adv_image = args.patch_applier(image_org, adv_patch_cpu)
        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join('../_output_img', args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            if 'JPEG' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + 'png')
            elif 'png' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name))



















