'''

Generate Generalized Images from trained networks
Reference: https://github.com/NVlabs/stylegan3
Reference: https://github.com/chi0tzp/WarpedGANSpace

'''

import click
import os
import time
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import random
from typing import List, Optional, Tuple, Union
import pickle
from metrics import metric_main

import legacy
import pdb
import shutil
import metrics
import copy

#----------------------------------------------------------------------------

def setup_images_label(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    #pdb.set_trace()
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]
    
    #pdb.set_trace()
    # Load data.
    images1, images2, labels1, labels2 = zip(*[training_set[i] for i in grid_indices])
    return np.stack(images1), np.stack(images2), np.stack(labels1), np.stack(labels2)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

def run_SP(SP, img, z_org, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets):
        #pdb.set_trace()
        #trg_support_sets_indices = torch.randint(0, num_support_sets, [img.shape[0]]).to(device=z_org.device)
        shift_magnitudes_pos = (min_shift_magnitude - max_shift_magnitude) * torch.rand(trg_support_sets_indices.size()) + max_shift_magnitude
        shift_magnitudes_neg = (min_shift_magnitude - max_shift_magnitude) * torch.rand(trg_support_sets_indices.size()) - min_shift_magnitude
        shift_magnitudes_pool = torch.cat((shift_magnitudes_neg, shift_magnitudes_pos))

        shift_magnitudes_ids = torch.arange(len(shift_magnitudes_pool), dtype=torch.float)
        trg_shift_magnitudes = shift_magnitudes_pool[torch.multinomial(input=shift_magnitudes_ids,
                                                                              num_samples=img.shape[0],
                                                                              replacement=False)]
        trg_shift_magnitudes = trg_shift_magnitudes.to(device=trg_support_sets_indices.device)
        #pdb.set_trace()
        supp_sets_mask = torch.zeros([img.shape[0], num_support_sets]).to(device=trg_support_sets_indices.device)
        for i, (index, val) in enumerate(zip(trg_support_sets_indices, trg_shift_magnitudes)):
                supp_sets_mask[i][index] += 1.0

        #pdb.set_trace()
        shift = trg_shift_magnitudes.reshape(-1, 1) * SP(supp_sets_mask, z_org)
        z_shift = torch.add(z_org, shift)
        z_shift = torch.nn.functional.normalize(z_shift, p=2, dim=1)
        #z_shift = z_org
        
        return z_shift, trg_shift_magnitudes

#----------------------------------------------------------------------------

def testing_loop(
    run_dir                 = '.',      # Output directory.
    testing_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    ES_kwargs               = {},       # Options for ES network.
    ED_kwargs               = {},       # Options for ED network.
    RC_kwargs               = {},
    SP_kwargs               = {},
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    ES_opt_kwargs           = {},       # Options for ES optimizer.
    ED_opt_kwargs           = {},       # Options for ED optimizer.
    RC_opt_kwargs           = {},       # Options for ED optimizer.
    SP_opt_kwargs           = {},       # Options for ED optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 12,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 1,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 1,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    ES_reg_interval         = None,     # How often to perform regularization for ES? None = disable lazy regularization.
    ED_reg_interval         = None,     # How often to perform regularization for ED? None = disable lazy regularization.
    RC_reg_interval         = None,     # How often to perform regularization for ED? None = disable lazy regularization.
    SP_reg_interval         = None,     # How often to perform regularization for ED? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 1,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 1,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 1,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = False,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    use_es                  = False,     # Use Style Encoder
    use_ed                  = False,     # Use Identity Encoder
    use_warp                = False,
    first_enc               = False,
    num_support_sets        = 128,
    min_shift_magnitude     = 0.25,
    max_shift_magnitude     = 0.45,
    truncation_psi          = 1
):
    # Initialize.
    start_time = time.time()

    os.makedirs(run_dir, exist_ok=True)


    device = torch.device('cpu', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load testing set.
    print('Loading testing set...')
    testing_set = dnnlib.util.construct_class_by_name(**testing_set_kwargs) # subclass of training.dataset.Dataset
    testing_set_sampler = misc.InfiniteSampler(dataset=testing_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    testing_set_iterator = iter(torch.utils.data.DataLoader(dataset=testing_set, sampler=testing_set_sampler, batch_size=1, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(testing_set))
        print('Image shape:', testing_set.image_shape)
        print('Label shape:', testing_set.label_shape)
        print()

    
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(use_es=use_es, use_ed=use_ed, c_dim=12, img_resolution=256, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    if use_es:
        #print('Build ES')
        ES = dnnlib.util.construct_class_by_name(**ES_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    else:
        ES = None
    
    if use_ed:
        #print('Build ED')
        ED = dnnlib.util.construct_class_by_name(**ED_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    else:
        ED = None
    
    if use_warp:
        SP = dnnlib.util.construct_class_by_name(**SP_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    else:
        SP = None
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    #pdb.set_trace()
    print('Loading networks from "%s"...' % resume_pkl)
    with dnnlib.util.open_url(resume_pkl) as f:
        data = legacy.load_network_pkl(f) # type: ignore
        #mdd = [('ES', ES), ('ED', ED), ('SP', SP), ('G_ema', G)]
        mdd = [('G_ema', G_ema)]
        for name, module in mdd:
            misc.copy_params_and_buffers(data[name], module, require_all=False)
    
    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))

    # Setup training phases.
    if rank == 0:
        print('Setting up testing phases...')
    
    #pdb.set_trace()

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    
    
    images1, images2, labels1, labels2 = setup_images_label(training_set=testing_set)
    
    #pdb.set_trace()
    for seed in range(0,2000):
        real_img1, real_img2, real_c1, real_c2 = next(testing_set_iterator)
        real_img1 = (real_img1.to(device).to(torch.float32) / 127.5 - 1)
        real_img2 = (real_img2.to(device).to(torch.float32) / 127.5 - 1)
        real_c1 = real_c1.to(device)
        real_c2 = real_c2.to(device)

        z1 = torch.randn([1, G_ema.z_dim//2], device=device)
        z2 = torch.randn([1, G_ema.z_dim//2], device=device)

        if use_es:
            z1 = ES(real_img1, real_c1)

        if use_ed:
            z2 = ED(real_img2, z2)
        
        if use_warp:
            trg_support_sets_indices = torch.randint(0, num_support_sets, [1], device=device)
            shift, _ = run_SP(SP, real_img2, z2, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets)
            shift = shift.to(device)
            zshift = torch.add(z2, shift).to(device)
            z2 = zshift
        
        z = torch.cat((z1, z2), dim=1).to(device)
        #z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        
        # Generalize generation
        # Generation with given styles
        #translates = [[0,0], [0.08,0.09], [0.05,0.03], [0.02,0.02], [0.045,0.045]]
        translates = [[0,0], [0.08,0.09], [0.05,0.03]]
        angles = [0,15,10]

        count = 0
        #outputs_and_filenames1 = []
        #outputs_and_filenames2 = []
        for translate in translates:
            for rotate in angles:

                if hasattr(G_ema.synthesis, 'input'):
                    m = make_transform(translate, rotate)
                    m = np.linalg.inv(m)
                    G_ema.synthesis.input.transform.copy_(torch.from_numpy(m))

                img = G_ema(z, real_c1, truncation_psi=truncation_psi, noise_mode='const')
                #_, y = D(real_img1, label)
                #_, y_hat = D(img, label)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{run_dir}/generate_{seed:04d}_{count:04d}.png')
                
                print("Subject: {} and Image: {}".format(seed, count))
                count += 1
    
    #with open('real_input_test.pkl', 'wb') as f:
    #    pickle.dump(outputs_and_filenames1, f)
    
    #with open('out_test.pkl', 'wb') as f:
    #    pickle.dump(outputs_and_filenames2, f)


#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str, required=True)
@click.option('--use_es',       help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--use_ed',       help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--use_warp',       help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), default='stylegan3-t')
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), default=1)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), default=1)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), default=8.2)

@click.option('--reconstructor-type', type=str, default='LeNet', help='set reconstructor network type')
@click.option('--min-shift-magnitude', type=float, default=0.25, help="set minimum shift magnitude")
@click.option('--max-shift-magnitude', type=float, default=0.45, help="set shifts magnitude scale")
@click.option('--reconstructor-lr', type=float, default=1e-4, help="set learning rate for reconstructor R optimization")
@click.option('-K', '--num-support-sets', type=int, default=128, help="set number of support sets (warping functions)")
@click.option('-D', '--num-support-dipoles', type=int, default=4, help="set number of support dipoles per support set")
@click.option('--learn_alphas', type=bool, default=True, help='learn RBF alpha params')
@click.option('--learn_gammas', type=bool, default=False, help='learn RBF gamma params')
@click.option('-g', '--gamma', type=float, default=0.001, help="set RBF gamma param; when --learn-gammas is set, this will be the initial value of gammas for all RBFs")
@click.option('--support-set-lr', type=float, default=1e-4, help="set learning rate")
@click.option('--lambda-cls', type=float, default=1.00, help="classification loss weight")
@click.option('--lambda-reg', type=float, default=0.25, help="regression loss weight")

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1), default=1)
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--support_lr',   help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--reconstructor_lr',          help='D learning rate', metavar='FLOAT',           type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=24, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)


def main(**kwargs):
    """
    Test the iWarpGAN
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Testing set.
    c.testing_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.testing_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.testing_set_kwargs.use_labels = opts.cond

    c.ED_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.IDNetwork', w_dim=512, z_dim=512, 
                                    num_support_sets = opts.num_support_sets, num_support_dipoles = opts.num_support_dipoles,
                                    learn_alphas = opts.learn_alphas, learn_gammas = opts.learn_gammas, 
                                    reconstructor_type = opts.reconstructor_type
                                )
    c.RC_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Reconstructor', w_dim=512, z_dim=512, 
                                    num_support_sets = opts.num_support_sets, num_support_dipoles = opts.num_support_dipoles,
                                    learn_alphas = opts.learn_alphas, learn_gammas = opts.learn_gammas, 
                                    reconstructor_type = opts.reconstructor_type
                                )
    c.SP_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.SupportSets', w_dim=512, z_dim=512, 
                                    num_support_sets = opts.num_support_sets, num_support_dipoles = opts.num_support_dipoles,
                                    learn_alphas = opts.learn_alphas, learn_gammas = opts.learn_gammas, 
                                    reconstructor_type = opts.reconstructor_type, gamma=opts.gamma
                                )
    c.ES_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.StyleNetwork', w_dim=512, z_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), 
                                    mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.ES_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.ED_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.RC_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.SP_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    c.use_es = opts.use_es
    c.use_ed = opts.use_ed
    c.use_warp = opts.use_warp

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (2 if opts.cfg == 'stylegan2' else 8) if opts.map_depth is None else opts.map_depth
    c.ES_kwargs.mapping_kwargs.num_layers = (2 if opts.cfg == 'stylegan2' else 8) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.0025 if opts.cfg == 'stylegan2' else 0.002) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.ES_opt_kwargs.lr = opts.dlr
    c.ED_opt_kwargs.lr = opts.dlr
    c.RC_opt_kwargs.lr = opts.reconstructor_lr
    c.SP_opt_kwargs.lr = opts.support_lr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.testing_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
    c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
    if opts.cfg == 'stylegan3-r':
        c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
        c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
        c.G_kwargs.channel_max *= 2
        c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
        c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
        c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
        
    
    testing_loop(run_dir=opts.outdir, **c)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
