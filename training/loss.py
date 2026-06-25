'''
Reference: https://github.com/NVlabs/stylegan3
Reference: https://github.com/chi0tzp/WarpedGANSpace
'''

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import math



#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, ES=None, ED=None, RC=None, SP=None, use_es = False, use_ed = False, use_warp=False, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.ES                 = ES
        self.ED                 = ED
        self.G                  = G
        self.D                  = D
        self.RC                 = RC
        self.SP                 = SP
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.use_es             = use_es
        self.use_ed             = use_ed
        self.use_warp           = use_warp
        self.loss_fn = torch.nn.MSELoss()
        self.loss_fn2 = torch.nn.BCELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
    
    # Define the transform function
    def warp_images(self, images, labels):
        #pdb.set_trace()
        batch_size, num_channels, height, width = images.size()

        # Extract angle and translation values from labels
        angle_values = labels[:, 0:5] * torch.tensor([0, 10, 12, 15, 18], device=self.device)
        
        translation_indices = torch.argmax(labels[:, 5:], dim=1)  # Get the indices of selected translations
        translation_values = labels[:, 5:10] * torch.tensor([1,2,3,4,5], device=self.device)
        # Define the available translation values
        translations = torch.tensor([[0, 0], [-10, -10], [15, 15], [-10, 15], [20, -10]], device=self.device)
        # Select the translation values based on the indices
        translation_values = translations[translation_indices]

        # Generate the transformation grid
        theta = torch.zeros(batch_size, 2, 3)
        theta[:, 0, 0] = torch.cos(angle_values[:, 0] * math.pi / 180.0)  # Convert angles to radians
        theta[:, 0, 1] = -torch.sin(angle_values[:, 0] * math.pi / 180.0)
        theta[:, 1, 0] = torch.sin(angle_values[:, 0] * math.pi / 180.0)
        theta[:, 1, 1] = torch.cos(angle_values[:, 0] * math.pi / 180.0)
        theta[:, :, 2] = translation_values

        grid = torch.nn.functional.affine_grid(theta, images.size()).to(self.device)

        # Warp the images using the grid
        warped_images = torch.nn.functional.grid_sample(images, grid)

        return warped_images
    
    def run_ES(self, img, c):
        z = self.ES(img, c)
        return z
    
    def run_ED(self, img, x2):
        #pdb.set_trace()
        z = self.ED(img, x2)
        return z
    
    def run_SP(self, img, z_org, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets):
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
        shift = trg_shift_magnitudes.reshape(-1, 1) * self.SP(supp_sets_mask, z_org)
        z_shift = torch.add(z_org, shift)
        z_shift = torch.nn.functional.normalize(z_shift, p=2, dim=1)
        #z_shift = z_org
        
        return z_shift, trg_shift_magnitudes
    
    def run_RC(self, img1, img2):
        #pdb.set_trace()
        predicted_support_sets_indices, predicted_shift_magnitudes = self.RC(img1, img2)
        
        return predicted_support_sets_indices, predicted_shift_magnitudes

    def run_G(self, z, c, update_emas=False):
        #pdb.set_trace()
        #z = torch.nn.functional.normalize(z, p=2, dim=1)
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits, pred_y = self.D(img, c, update_emas=update_emas)
        return logits, pred_y

    def accumulate_gradients(self, phase, real_img1, real_img2, real_c1, real_c2, gen_z1, gen_z2, gen_c, gain, cur_nimg, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets, use_es=True, use_ed=True, use_warp=False):
        assert phase in ['G_ESmain', 'G_ESreg', 'G_ESboth', 
                        'G_EDmain', 'G_EDreg', 'G_EDboth', 
                        'RCmain', 'RCreg', 'RCboth', 
                        'SPmain', 'SPreg', 'SPboth', 
                        'Gmain', 'Greg', 'Gboth', 
                        'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            #phase = {'G_ESreg': 'none', 'G_ESboth': 'G_ESmain', 'Greg': 'none', 'Gboth': 'Gmain', 'ESmain': 'none', 'EDmain': 'none', 'RCmain': 'none'}.get(phase, phase)
            phase = {'G_ESreg': 'none', 'G_ESboth': 'G_ESmain', 'G_EDreg': 'none', 'G_EDboth': 'G_EDmain', 'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        #pdb.set_trace()
        
        if phase in ['RCmain', 'RCboth', 'RCreg', 'SPmain', 'SPboth', 'SPreg']:
            #pdb.set_trace()
            with torch.autograd.profiler.record_function('RCmain_forward'):
                z1 = gen_z1
                z2 = gen_z2
                
                if use_es:
                    #z1 = self.run_ES(real_img1, gen_c)
                    z1 = self.run_ES(real_img1, real_c1)
                if use_ed:
                    #z2 = self.run_ED(real_img2, gen_c)
                    z2 = self.run_ED(real_img2, z2)
                z = torch.cat((z1, z2), dim=1)

                z_shift, trg_shift_magnitudes = self.run_SP(real_img2, z2, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets)
                z_shifted = torch.cat((z1, z_shift), dim=1)
                
                #gen_img_org, _ = self.run_G(z, gen_c)
                gen_img_org, _ = self.run_G(z, real_c1)
                #gen_img_shifted, _ = self.run_G(z_shifted, gen_c)
                gen_img_shifted, _ = self.run_G(z_shifted, real_c1)

                predicted_support_sets_indices, predicted_shift_magnitudes = self.run_RC(gen_img_org, gen_img_shifted)

                #pdb.set_trace()

                classification_loss = self.cross_entropy(predicted_support_sets_indices, trg_support_sets_indices)
                regression_loss = torch.mean(torch.abs(predicted_shift_magnitudes - trg_shift_magnitudes))

                training_stats.report('Loss/RC/loss', regression_loss)
            with torch.autograd.profiler.record_function('RCmain_backward'):
                loss = classification_loss + 0.25 * regression_loss
                loss.mean().backward()
            
        if phase in ['Gmain', 'Gboth', 'G_ESmain', 'G_ESboth', 'G_EDmain', 'G_EDboth']:
            #pdb.set_trace()
            with torch.autograd.profiler.record_function('Gmain_forward'):
                z1 = gen_z1
                z2 = gen_z2
                
                if use_es:
                    #z1 = self.run_ES(real_img1, gen_c)
                    z1 = self.run_ES(real_img1, real_c1)
                if use_ed:
                    #z2 = self.run_ED(real_img2, gen_c)
                    z2 = self.run_ED(real_img2, z2)
                #pdb.set_trace()
                z = torch.cat((z1, z2), dim=1)

                if use_warp:
                    z_shift, trg_shift_magnitudes = self.run_SP(real_img2, z2, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets)
                    z = torch.cat((z1, z_shift), dim=1)
                
                #gen_img, _gen_ws = self.run_G(z, gen_c)
                gen_img, _gen_ws = self.run_G(z, real_c1)
                if use_es:
                    #z_synth1 = self.run_ES(gen_img, gen_c)
                    #warp_img = self.warp_images(gen_img, gen_c)
                    z_synth1 = self.run_ES(gen_img, real_c1)
                    warp_img = self.warp_images(gen_img, real_c1)
                if use_ed:
                    #z_synth2 = self.run_ED(gen_img, gen_c)
                    z_synth2 = self.run_ED(gen_img, z2)
                #gen_logits, y_pred = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                gen_logits, y_pred = self.run_D(gen_img, real_c1, blur_sigma=blur_sigma)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                if use_es and use_ed:
                    loss_ESmain = self.loss_fn(z1, z_synth1)
                    #pred_loss = self.loss_fn2(y_pred, gen_c)
                    pred_loss = self.loss_fn2(y_pred, real_c1)
                    loss_EDmain = -self.loss_fn(z2, z_synth2)
                    loss = loss_Gmain + 0.1*loss_ESmain + 0.1*loss_EDmain
                elif use_es:
                    loss_ESmain = self.loss_fn(z1, z_synth1)
                    #pred_loss = self.loss_fn2(y_pred, gen_c)
                    pred_loss = self.loss_fn2(y_pred, real_c1)
                    loss_sty = self.loss_fn(gen_img, warp_img)
                    loss = loss_Gmain + 0.1*loss_ESmain + 0.1*loss_sty
                elif use_ed:
                    loss_EDmain = -self.loss_fn(z2, z_synth2)
                    loss = loss_Gmain + 0.1*loss_EDmain
                else:
                    loss  = loss_Gmain

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss.mean().mul(gain).backward()


        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth', 'G_ESreg', 'G_ESboth', 'G_EDreg', 'G_EDreg']:
            #pdb.set_trace()
            with torch.autograd.profiler.record_function('Gpl_forward'):
                z1 = gen_z1
                z2 = gen_z2
                
                if use_es:
                    #z1 = self.run_ES(real_img1, gen_c)
                    z1 = self.run_ES(real_img1, real_c1)
                if use_ed:
                    #z2 = self.run_ED(real_img2, gen_c)
                    z2 = self.run_ED(real_img2, z2)
                z = torch.cat((z1, z2), dim=1)

                if use_warp:
                    z_shift, trg_shift_magnitudes = self.run_SP(real_img2, z2, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets)
                    z = torch.cat((z1, z_shift), dim=1)

                batch_size = z.shape[0] // self.pl_batch_shrink
                #gen_img, gen_ws = self.run_G(z[:batch_size], gen_c[:batch_size])
                gen_img, gen_ws = self.run_G(z[:batch_size], real_c1[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)

            with torch.autograd.profiler.record_function('Gpl_backward'):
                (loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            #pdb.set_trace()
            with torch.autograd.profiler.record_function('Dgen_forward'):
                z1 = gen_z1
                z2 = gen_z2
                if use_es:
                    #z1 = self.run_ES(real_img1, gen_c)
                    z1 = self.run_ES(real_img1, real_c1)
                if use_ed:
                    #z2 = self.run_ED(real_img2, gen_c)
                    z2 = self.run_ED(real_img2, z2)
                z = torch.cat((z1, z2), dim=1)

                if use_warp:
                    z_shift, trg_shift_magnitudes = self.run_SP(real_img2, z2, min_shift_magnitude, max_shift_magnitude, trg_support_sets_indices, num_support_sets)
                    z = torch.cat((z1, z_shift), dim=1)
                
                #gen_img, _gen_ws = self.run_G(z, gen_c, update_emas=True)
                #gen_logits, y_pred = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)

                gen_img, _gen_ws = self.run_G(z, real_c1, update_emas=True)
                gen_logits, y_pred = self.run_D(gen_img, real_c1, blur_sigma=blur_sigma, update_emas=True)
                
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                try:
                    y_pred = y_pred.reshape(real_c1.shape)
                except:
                    pass
                pred_loss = self.loss_fn2(y_pred, real_c1)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                ((loss_Dgen + 0.1* pred_loss).mean().mul(gain)).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            #pdb.set_trace()
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img1.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits, y_pred = self.run_D(real_img_tmp, real_c1, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
                
                pred_loss = self.loss_fn2(y_pred, real_c1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                ((loss_Dreal + loss_Dr1 + 0.1 * pred_loss).mean().mul(gain)).backward()

#----------------------------------------------------------------------------
