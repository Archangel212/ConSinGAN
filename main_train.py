import datetime
import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
import random

from ConSinGAN.config import get_arguments
from ConSinGAN.manipulate import *
import ConSinGAN.functions as functions


def get_scale_factor(opt):
    opt.scale_factor = 1.0
    num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    opt.scale_factor_init = opt.scale_factor
    if opt.num_training_scales > 0:
        while num_scales > opt.num_training_scales:
            opt.scale_factor_init = opt.scale_factor_init - 0.01
            num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    return opt.scale_factor_init


# noinspection PyInterpreter
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--sample', action='store_true', help='generate random samples', default=0)
    parser.add_argument('--timestamp', help='task to be done', default='train')
    parser.add_argument('--train_depth', type=int, help='how many layers are trained if growing', default=3)
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for layers if growing', default=0.1)
    parser.add_argument('--start_scale', type=int, help='at which scale to start training', default=0)
    parser.add_argument('--train_scales', type=int, help='at which scale to start training', default=3)
    parser.add_argument('--hc_scales', action='store_true', help='hard coded scales', default=0)
    parser.add_argument('--harmonization_img', help='for harmonization', type=str, default='')
    parser.add_argument('--add_img', action='store_true', help='use augmented img for adversarial loss', default=0)
    parser.add_argument('--fine_tune', action='store_true', help='fine tune on final image', default=0)
    parser.add_argument('--fine_tune_model', action='store_true', help='fine tune on final image', default=0)
    parser.add_argument('--model_finetune_dir', help='input image name', required=False)
    parser.add_argument('--hq', action='store_true', help='fine tune on high res image', default=0)
    parser.add_argument('--add_mask', action='store_true', help='fine tune on high res image', default=0)
    parser.add_argument('--num_training_scales', type=int, help='how many scales to train on', default=0)
    parser.add_argument('--edit_add_noise', action='store_true', help='fine tune on high res image', default=0)

    parser.add_argument('--batch_norm', action='store_true', help='"use batch norm in generator"', default=0)


    opt = parser.parse_args()
    _timestamp = opt.timestamp
    opt = functions.post_config(opt)

    if opt.ProSinGAN:
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            from SinGAN.training_prosingan_generation import *
        elif opt.train_mode == "harmonization":
            if opt.fine_tune_model:
                if opt.hq:
                    from SinGAN.training_prosingan_harmonization_finetune_model_highres import *
                else:
                    from SinGAN.training_prosingan_harmonization_finetune_model import *
            else:
                from SinGAN.training_prosingan_harmonization import *
        elif opt.train_mode == "editing":
            if opt.fine_tune_model:
                from SinGAN.training_prosingan_editing_finetune_model import *
            else:
                from SinGAN.training_prosingan_editing import *
    elif opt.MSGGan:
        from SinGAN.training_msggan import *
    elif opt.addDs:
        from SinGAN.training_addDs import *
    else:
        from SinGAN.training import *

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    if opt.train_multiple_images:
        opt.imgs = glob.glob(opt.input_dir + "/" + opt.input_name + "*")

    torch.cuda.set_device(opt.gpu)
    real = functions.read_image(opt)

    if opt.train_mode == "generation" or opt.train_mode == "retarget":
        if opt.num_training_scales > 0:
            opt.scale_factor_init = get_scale_factor(opt)
    if opt.train_mode == "original":
        if opt.num_training_scales > 0:
            opt.scale_factor_init = get_scale_factor(opt)

    opt.scale_factor = opt.scale_factor_init

    ################
    # image = np.ones((300, 300, 3), dtype=np.uint8)
    # image = img.imread('%s/%s' % (opt.input_dir,opt.input_name))#.resize((202,250))
    # img.imsave("test_imgs/real_img.png", image)
    #
    # data = {"image": image}
    # aug = Augment()
    # for idx in range(50):
    #
    # print(image.shape)
    # exit()

    # # transorms = functions.ImageTransforms()
    # real = img.imread('%s/%s' % (opt.input_dir,opt.input_name))#.resize((202,250))
    # print(real.shape)
    # # real = imresize_to_shape(real, [202, 250], opt)
    # # real = real.cpu().numpy().squeeze().transpose(1,2,0)
    # # print(real[0, :5, 0])
    # # exit()
    #
    # print(real.shape)
    # t_img = transorms.transform(**{"image": real})
    # print(t_img["real"].shape)
    # exit()

    ################

    dir2save = functions.generate_dir2save(opt)

    if osp.exists(dir2save):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        with open(osp.join(dir2save, 'opt.txt'), 'w') as f:
            for o in opt.__dict__:
                f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
        current_path = os.path.dirname(os.path.abspath(__file__))
        for py_file in glob.glob(osp.join(current_path, "*.py")):
            copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
        copytree(osp.join(current_path, "SinGAN"), osp.join(dir2save, "SinGAN"))
        copytree(osp.join(current_path, "SIFID"), osp.join(dir2save, "SIFID"))

        print("Training model: {} - {}".format(opt.timestamp, opt.note))
        functions.adjust_scales2image(real, opt)

        start = time.time()
        if opt.noise_norm:
            x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
            x = x[:, :, :3].reshape(-1)
            x_grid = np.linspace(0, 255, 256)
            opt.x_grid = x_grid
            kde_pdf = gaussian_kde(x, bw_method=0.1 / x.std(ddof=1))
            kde_pdf = kde_pdf.evaluate(x_grid)
            cdf = np.cumsum(kde_pdf)
            cdf = cdf / cdf[-1]
            opt.cdf = cdf

        train(opt, Gs, Zs, reals, NoiseAmp)
        end = time.time()
        elapsed_time = end - start
        print("Time for training: {} seconds".format(elapsed_time))
        # print(type(elapsed_time))
        # print("{}/time_{:.3f}.txt".format(dir2save, elapsed_time))
        with open("{}/time_{:.3f}.txt".format(dir2save, elapsed_time), "w") as f:
            f.write("Time: {} seconds.".format(elapsed_time))
        print(end - start)
        if not opt.ProSinGAN and not opt.addDs and not opt.MSGGan:
            opt.mode = 'random_samples'
            opt.gen_start_scale = 0
            SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)