import os
import json
import copy
import time
import logging
import argparse
import pickle
import numpy as np
import torch as th
import torch.distributed as dist
from tqdm import tqdm

th.manual_seed(0)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util_transformer import (
    create_model_condition_and_diffusion,
)
from utils import utils_transform
from utils import utils_option as option

from utils import utils_logger

from itertools import product
from collections import OrderedDict
from human_body_prior.body_model.body_model import BodyModel

from data.dataset_amass import AMASS_ALL_Dataset
from torch.utils.data import DataLoader
from data.select_dataset import define_Dataset
from models.select_model import define_Model

from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa,global2local_pose
from guided_diffusion.respace import space_timesteps

from Samplers import *
from PostProcessing import *
from Measurement import *

from functorch import vmap

def create_opts():

    # args = parser.parse_args()
    json_str = ''
    with open('options/test.json', 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    opt = option.parse('options/test.json', "1", is_train=True)
    opt['diffusion']['timestep_respacing'] = 'ddim50'
    opt['use_ddim'] = True
    opt['clip_denoised'] = True
    opt['num_evaluation'] = 1
    opt['save_vid'] = 0
    opt['steps'] = 1
    opt['identifier'] = ''
    opt['gpu_ids'] = '0'

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    return opt


opt = create_opts()

print(opt['path']['pretrained'])

logger_name = 'test'
utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
logger_transformer = logging.getLogger(logger_name)

dist_util.setup_dist(devices=opt['gpu_ids'])
logger.configure(dir=opt['path']['root'])

logger.log("creating BoDiffusion...")

# ----------------------------------------
# 1) create_dataset
# 2) creat_dataloader for train and test
# ----------------------------------------
dataset_type = opt['datasets']['test']['dataset_type']
for phase, dataset_opt in opt['datasets'].items():

    if phase == 'test':
        test_set = define_Dataset(dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1,  # dataset_opt['dataloader_batch_size'],
                                 shuffle=False, num_workers=1,
                                 drop_last=False, pin_memory=True)
    elif phase == 'train':
        continue
    else:
        raise NotImplementedError("Phase [%s] is not recognized." % phase)


# Model Initialization
model, diffusion = create_model_condition_and_diffusion(
    use_fp16=opt['fp16']['use_fp16'],
    **opt['ddpm'],
    **opt['diffusion']
)
chpnt = opt['path']['pretrained']
print(f'resuming checkpoint from {chpnt}')
model.load_state_dict(
    dist_util.load_state_dict(
        os.path.join(opt['path']['pretrained']),
        map_location="cpu"
    )
)
model.to(dist_util.dev())
if opt['fp16']['use_fp16']:
    model.convert_to_fp16()
model.eval()

logger.log("sampling...")
all_images = []
all_labels = []
shape = (
    opt['datasets']['test']['dataloader_batch_size'],
    opt['ddpm']['in_channels'],
    *opt['ddpm']['image_size']
)

# instantiate the BodyModel
subject_gender = "male"
bm_fname = os.path.join(opt['support_dir'], 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = os.path.join(opt['support_dir'], 'body_models/dmpls/{}/model.npz'.format(subject_gender))
num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters
body_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(dist_util.dev())

parents = body_model.kintree_table[0][:22].long().clone().cpu()

bs = opt['datasets']['test']['dataloader_batch_size']
idx = 0

error_stats = {
    'rot_error': [],
    'pos_error': [],
    'vel_error': [],
    'pos_error_hands': [],
    'rot_error_hands_and_head': [],
    'pos_error_upper': [],
    'pos_error_lower': [],
    'pos_error_pelvis': [],
}
logger.log('Evaluating {} times per timestep'.format(opt['num_evaluation']))

dist.barrier()

# ============================================================================================================
# COMPUTE FID AND OTHER METRICS FROM INSTANCES
# ============================================================================================================

window_slide = 20
test_times = [999]
if opt['diffusion']['timestep_respacing'] != '':
    if opt['diffusion']['timestep_respacing'].startswith("ddim"):
        timestep_respacing = int(opt['diffusion']['timestep_respacing'][len("ddim") :])
    else:
        timestep_respacing = opt['diffusion']['timestep_respacing']
    test_times = [int(i * int(timestep_respacing) / int(opt['diffusion']['diffusion_steps'])) for i in test_times]

# load amass dataset
dataset = AMASS_ALL_Dataset(opt=opt['datasets']['test'])
loader = th.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=10,
    drop_last=True
)

print(test_times)


joint_names = ["root", "left hip", "right hip", "abdomen", "left knee", "right knee", "lower rib", "left heel", "right heel", 
                   "upper rib", "left toe", "right toe", "neck", "left shoulder blade", "right shoulder blade", "face", "left shoulder", 
                   "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist"]

bones_list = [[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
              [1.,1.,1.,1.,1.,1.,0.85,1.,1.,0.85,1.,1.,0.85,0.85,0.85,0.85,0.85,0.85,0.85,0.85,0.85,0.85],
             [1.,1.,1.,1.,1.,1.,1.17,1.,1.,1.17,1.,1.,1.17,1.17,1.17,1.17,1.17,1.17,1.17,1.17,1.17,1.17]]

experiments = ['_default', '_upper_0.85', '_upper_1.17']

def main():

    dir_count = str(len(next(os.walk('experiments'))[1]))
    directory = "experiments/run_" + dir_count
    os.mkdir(directory)

    for alg in ['pigdm', 'cfg']:

        # for m in [1.4,1.2,0.8,0.6,0.7,1.3]:
        for exp_idx, exp in enumerate(experiments):

            # bones = None
            bones = bones_list[exp_idx]
    
            # with open(directory + "/exp_" + alg + "_mult_" + str(m) + ".txt", "w") as file:
            with open(directory + "/exp_" + alg + str(exp) + ".txt", "w") as file:
    
                # multiple = m
                multiple = 1.
        
                results = {i: [copy.deepcopy(error_stats) for _ in range(opt['num_evaluation'])] for i in test_times}
                pr_arr = {i: [] for i in test_times}
                rd_arr = {i: [] for i in test_times}
                max_ws = max(opt['datasets']['test']['window_size'], opt['datasets']['test']['cond_window_size'])
                ws = opt['datasets']['test']['window_size']
                cws = opt['datasets']['test']['cond_window_size']
                
                if not opt['diffusion']['timestep_respacing']:
                    respacing_mode = test_times
                else: 
                    respacing_mode = opt['diffusion']['timestep_respacing']
                sampling_times = space_timesteps(opt['diffusion']['diffusion_steps'], respacing_mode)
                sampling_times = sorted(sampling_times)
                
                start_count = 0
                max_idx = 138
                
                pos_errors_sample = {i: [] for i in range(start_count, max_idx)}
                rot_errors_sample = {i: [] for i in range(start_count, max_idx)}
                vel_errors_sample = {i: [] for i in range(start_count, max_idx)}
                l_pos_errors_sample = {i: [] for i in range(start_count, max_idx)}
                u_pos_errors_sample = {i: [] for i in range(start_count, max_idx)}
                
                for i in range(opt['num_evaluation']):
                    start = time.time()
                    for steps in test_times:
                        vidx = 0
                        rot_error = []
                        pos_error = []
                        error_dir = os.path.join(opt['path']['task'],'errors',opt['identifier'])
                        # if not os.path.exists(error_dir):
                        #     os.makedirs(error_dir)   
                        preds_dir = os.path.join(opt['path']['task'],'preds',opt['identifier'])
                        # if not os.path.exists(preds_dir):
                        #     os.makedirs(preds_dir)         
                        count = 0
                        for idx, test_data in enumerate(test_loader):
                
                            overlap = 0
                
                            if count < start_count:
                                count += 1
                                continue
                                
                            if count >= max_idx:
                                break
                                
                            if test_data['H'].shape[1] < max_ws:
                                continue
                
                            rot_error = 0.0 # 0.085 - 5 degrees
                
                            cond_full = test_data['L']
                            
                            # print(cond_full.shape)
                
                            cond_full[:,:,39:45] = 0.        # Only Head
                            cond_full[:,:,48:] = 0.
                            
                            # cond_full[:,:,36:] = 0.
                            cond_full[:,:,:36] += th.randn_like(cond_full[:,:,:36]) * rot_error
                            cond_full_clone = cond_full.clone().squeeze(0)
                            print('cond full = ' + str(cond_full_clone.shape))
                            cond_full = th.permute(cond_full.squeeze(0).reshape(-1, 3, 18), (2, 0, 1))
                            ground_truth_full = test_data['H']  # .squeeze(0)
                
                
                            if th.any(th.isnan(cond_full)):
                                print("cond_full contains NaN values.")
                            
                            Time = cond_full.size(1)
                            print(Time)
                
                            inner_iter = Time // (401-overlap) + 1
                
                            prev_window = None
                            for i_iter in range(inner_iter):
                                
                                th.cuda.empty_cache()
                
                                start_idx = i_iter * (401 - overlap)
                                end_idx = min(Time, start_idx + 401)
                                
                                if end_idx-start_idx < ws:
                                    break
                                    
                                num_windows = ((end_idx - start_idx - ws) // 20)
                                end_idx = start_idx + num_windows * 20 + 41
                
                                if overlap > end_idx - start_idx or i_iter > 10:
                                    break
                
                                print('iter length = ' + str(end_idx-start_idx))
                                
                                cond = th.nan_to_num(cond_full[:, start_idx:end_idx])
                                ground_truth = th.nan_to_num(ground_truth_full[:, start_idx:end_idx])
                                head_data = th.nan_to_num(test_data['Head_trans_global'][:, start_idx:end_idx].squeeze())
                                
                                cond = cond.to(dist_util.dev())
                                ground_truth = ground_truth.to(dist_util.dev())
                                head_data = head_data.to(dist_util.dev())
                
                                # Joint noise
                                joint_noise = 0.
                                
                                gt = final_prediction_local(ground_truth, head_data, body_model, True, dist_util.dev(), multiple=1.)       # For getting original translation
                                
                                new_head_trans = gt["trans"]
                                # print('gt trans val = ' + str(gt["trans"][10,:]))
                                
                                gt = final_prediction_local(ground_truth, head_data, body_model, True, dist_util.dev(), bones=bones, multiple=1.)        # For getting modified head translation
                
                                new_head_trans += gt["position_notran"][:,15,:]
                                head_data[:,:3,3] = new_head_trans
                                
                                
                                gt = final_prediction_local(ground_truth, head_data, body_model, True, dist_util.dev(), bones=bones, multiple=multiple)         # Final correct ground truth
                
                                #######################
                
                                # Manual conditioning
                
                                gt_position = gt['position'][:,[15,20,21],:].to(dist_util.dev())
                                
                                # Add noise
                                noise_std = 0.
                                gt_position += th.randn_like(gt_position) * noise_std
                                
                                gt_vel = gt_position[1:] - gt_position[:-1]
                                
                                gt_vel = th.concatenate([th.zeros((1,3,3), device=dist_util.dev()), gt_vel], dim=0)
                
                                cond_manual = th.nan_to_num(cond_full_clone[start_idx:end_idx,:].clone())
                                cond_manual[:,36:45] = gt_position.reshape(-1,9)
                                cond_manual[:,45:] = gt_vel.reshape(-1,9)
                
                                # For pigdm
                                
                                if alg == 'pigdm':
                                    cond_manual[:,39:45] = 0.          # Allow head
                                    cond_manual[:,48:54] = 0.          # Allow head
                
                                    # print('cond manual val = ' + str(cond_manual[10,36:39]))
                                    # cond_manual[:,36:39] /= multiple
                                    # cond_manual[:,45:48] /= multiple
                                
                                # cond_manual[:,36:] = 0.
                                cond_manual = th.permute(cond_manual.reshape(-1, 3, 18), (2, 0, 1)).to(dist_util.dev())
                
                                ########################
                
                                # PiGDM parameters
                                
                                num_windows =  1 + (end_idx-start_idx - ws) // window_slide
                                print('num_windows = ' + str(num_windows))
                                indexes = th.cat([th.arange(ws).view(1, -1)] * num_windows) + th.arange(num_windows).view(-1, 1) * window_slide
                
                                y = gt['position']
                                # Add noise
                                y += th.randn_like(y) * noise_std
                                y = y[:,[15,20,21],:]
                                # print('y = ' + str(y.shape))
                                y_batched = y[indexes,:,:]
                                y_batched_array = y_batched.swapaxes(2,3).reshape(num_windows,-1)
                                
                                A = A_system(parents, dist_util.dev(), j_noise=joint_noise, bones=bones, multiple=multiple)
                                
                                S = S_system(dist_util.dev())
                                
                                y_batched_array_s = (S @ y_batched_array.T).T
                
                                ##########################

                                if alg == 'cfg':
                    
                                    pred = sampling_cfg(
                                        diffusion,
                                        model,
                                        steps,
                                        cond_manual,  # conditioning sequence [18 x T x 3]
                                        window_slide=window_slide,
                                        ws=ws,
                                        device=dist_util.dev(),
                                        sampling_times=sampling_times,
                                    )

                                else:
                    
                                    # Without translation input
                                    
                                    pred = sampling_pigdm_batched(
                                        diffusion,
                                        model,
                                        steps,
                                        cond_manual,  # conditioning sequence [18 x T x 3]
                                        A=S@A,
                                        y=y_batched_array_s,
                                        grad_coeff=5,              # S - 5, V - 20
                                        window_slide=window_slide,
                                        ws=ws,
                                        device=dist_util.dev(),
                                        sampling_times=sampling_times,
                                    )
                                
                                # =======================================
                                # process all frames
                                # =======================================
                    
                                if th.any(th.isnan(pred)):
                                    print("pred contains NaN values.")
                
                                    break
                                    
                                pred = th.permute(pred, (1, 0, 2))
                                ground_truth = ground_truth.squeeze(0)
                                t = th.tensor([steps] * pred.size(0), device=pred.device)
                                # noised = diffusion.q_sample(pred.detach(), t, noise=th.randn_like(pred))
                                # Time = pred.size(0)
                
                    
                                pr = final_prediction(pred, head_data,body_model, device=pred.device, bones=bones, multiple=multiple)
                                gt = final_prediction_local(ground_truth, head_data,body_model, True, pred.device, bones=bones, multiple=multiple)
                
                                # For rendering                
                                # pr_verts = pr['body'].v
                
                                # if th.any(th.isnan(pr_verts)):
                                #     print("pr_verts contains NaN values.")
                
                    
                                # print(gt)
                    
                                gt_angle = gt['pose_body'].reshape(-1, 21, 3).detach()
                                pr_angle = pr['pose_body'].reshape(-1, 21, 3).detach()
                    
                                gt_pos = gt['position'].reshape(-1, 22, 3).detach()
                                pr_pos = pr['position'].reshape(-1, 22, 3).detach()
                    
                                gt_vel = (gt_pos[1:, ...] - gt_pos[:-1, ...]) * 60
                                pr_vel = (pr_pos[1:, ...] - pr_pos[:-1, ...]) * 60
                
                    
                                rot_error_ = th.nan_to_num(th.mean(th.absolute(gt_angle - pr_angle)).cpu()).numpy()
                                rot_error_hands_and_head_ = th.nan_to_num(th.mean(th.absolute(gt_angle - pr_angle)[:, [15 - 1, 20 - 1, 21 - 1], :]).cpu()).numpy()
                                pos_error_ = th.nan_to_num(th.mean(th.sqrt(th.sum(th.square(gt_pos - pr_pos), axis=-1))).cpu()).numpy()
                                pos_error_hands_ = th.nan_to_num(th.mean(th.sqrt(th.sum(th.square(gt_pos - pr_pos), axis=-1))[...,[20, 21]]).cpu()).numpy()
                                vel_error_ = th.nan_to_num(th.mean(th.sqrt(th.sum(th.square(gt_vel - pr_vel), axis=-1))).cpu()).numpy()
                                pos_error_upper_ = th.nan_to_num(th.mean(th.sqrt(th.sum(th.square(gt_pos-pr_pos), axis=-1))[...,[3,6,9,12,13,14,15,16,17,18,19,20,21]]).cpu()).numpy()
                                pos_error_lower_ = th.nan_to_num(th.mean(th.sqrt(th.sum(th.square(gt_pos-pr_pos), axis=-1))[...,[1,2,4,5,7,8,10,11]]).cpu()).numpy()
                                pos_error_pelvis_ = th.nan_to_num(th.mean(th.sqrt(th.sum(th.square(gt_pos-pr_pos), axis=-1))[...,[0]]).cpu()).numpy()
                                
                                pos_errors_sample[idx].append(pos_error_.copy())
                                rot_errors_sample[idx].append(rot_error_.copy())
                                u_pos_errors_sample[idx].append(pos_error_upper_.copy())
                                l_pos_errors_sample[idx].append(pos_error_lower_.copy())
                                vel_errors_sample[idx].append(vel_error_.copy())
                    
                                indv_errors = {
                                    'rot_error': rot_error_,
                                    'pos_error': pos_error_,
                                    'vel_error': vel_error_,
                                    'pos_error_hands': pos_error_hands_,
                                    # 'noise_rot_error': rot_error_nn_,
                                    'rot_error_hands_and_head': rot_error_hands_and_head_,
                                    'pos_error_upper': pos_error_upper_,
                                    'pos_error_lower': pos_error_lower_,
                                    'pos_error_pelvis': pos_error_pelvis_,
                    
                                }     
                                # with open(filename_, 'wb') as f:
                                #     pickle.dump(indv_errors, f)
                    
                                results[steps][i]['rot_error'].append(rot_error_)
                                results[steps][i]['pos_error'].append(pos_error_)
                                results[steps][i]['vel_error'].append(vel_error_)
                                results[steps][i]['pos_error_hands'].append(pos_error_hands_)
                                results[steps][i]['rot_error_hands_and_head'].append(rot_error_hands_and_head_)
                                # pr_arr[steps].append(pr['pose_body'].cpu().detach())
                                # rd_arr[steps].append(nn['pose_body'].cpu().detach())
                    
                                results[steps][i]['pos_error_upper'].append(pos_error_upper_)
                                results[steps][i]['pos_error_lower'].append(pos_error_lower_)
                                results[steps][i]['pos_error_pelvis'].append(pos_error_pelvis_)
                                
                                print('Iteration [{}] | Sample [{} / {}] | Steps [{}] | Time [{:<.5f}]'.format(i + 1, idx + 1, len(test_loader), steps, time.time() - start))
                                if idx % 10 == 0:
                                    # logger_transformer.info(f'rot_error: {sum(rot_error) / len(rot_error) * 57.2958}, pos_error: {sum(pos_error) / len(pos_error)*100}')
                                    logger_transformer.info(
                                    f"Average errors for iter {idx} steps {steps}"
                                    )
                                    logger_transformer.info(
                                    "Average rotational error [degree]: {:<.5f}, H+Hs rotational error [degree]: {:<.5f}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average positional error at hand [cm]: {:<.5f}, Avpos error at upper [cm]: {:<.5f}, Avpos error at lower [cm]: {:<.5f}, Avpos error at pelvis [cm]: {:<.5f}\n".format(
                                        sum(results[steps][i]['rot_error']) / len(results[steps][i]['rot_error']) * 57.2958, sum(results[steps][i]['rot_error_hands_and_head']) / len(results[steps][i]['rot_error_hands_and_head']) * 57.2958, sum(results[steps][i]['pos_error']) / len(results[steps][i]['pos_error']) * 100, sum(results[steps][i]['vel_error']) / len(results[steps][i]['vel_error']) * 100, sum(results[steps][i]['pos_error_hands']) / len(results[steps][i]['pos_error_hands']) * 100, sum(results[steps][i]['pos_error_upper']) / len(results[steps][i]['pos_error_upper']) * 100, sum(results[steps][i]['pos_error_lower']) / len(results[steps][i]['pos_error_lower']) * 100, sum(results[steps][i]['pos_error_pelvis']) / len(results[steps][i]['pos_error_pelvis']) * 100
                                    )
                                    )
                    
                                
                                del gt
                                del pr
                                
                                del gt_pos
                                del pr_pos
                                
                                del gt_vel
                                del pr_vel
                                
                                del gt_angle
                                del pr_angle
                
                                del pred
                    
                            count += 1
                
                    logger.log('Done evaluation [{}] | Steps [{}] | Time [{:<.5f}]'.format(i + 1, steps, time.time() - start))
                    logger.log('=' * 75)
                
                logger.log('Finished evaluation')
                
                for steps, i in product(test_times, range(opt['num_evaluation'])):
                    results[steps][i]['rot_error'] = sum(results[steps][i]['rot_error'])/len(results[steps][i]['rot_error'])
                    results[steps][i]['pos_error'] = sum(results[steps][i]['pos_error'])/len(results[steps][i]['pos_error'])
                    results[steps][i]['vel_error'] = sum(results[steps][i]['vel_error'])/len(results[steps][i]['vel_error'])
                    results[steps][i]['pos_error_hands'] = sum(results[steps][i]['pos_error_hands'])/len(results[steps][i]['pos_error_hands'])
                    results[steps][i]['rot_error_hands_and_head'] = sum(results[steps][i]['rot_error_hands_and_head'])/len(results[steps][i]['rot_error_hands_and_head'])
                    results[steps][i]['pos_error_upper'] = sum(results[steps][i]['pos_error_upper'])/len(results[steps][i]['pos_error_upper'])
                    results[steps][i]['pos_error_lower'] = sum(results[steps][i]['pos_error_lower'])/len(results[steps][i]['pos_error_lower'])
                    results[steps][i]['pos_error_pelvis'] = sum(results[steps][i]['pos_error_pelvis'])/len(results[steps][i]['pos_error_pelvis'])
                
                # compute final results
                final_results = {t: copy.deepcopy(error_stats) for t in test_times}
                # for steps, k in product(test_times, ['rot_error', 'pos_error', 'vel_error', 'pos_error_hands', 'noise_rot_error', 'rot_error_hands_and_head', 'noise_pos_error','noise_vel_error','pos_error_upper','pos_error_lower','pos_error_pelvis','pos_error_nn_hands','pos_error_nn_upper','pos_error_nn_lower','pos_error_nn_pelvis']):
                #     factor = 57.2958 if k in ['rot_error', 'noise_rot_error', 'rot_error_hands_and_head'] else 100
                
                for steps, k in product(test_times, ['rot_error', 'pos_error', 'vel_error', 'pos_error_hands', 'rot_error_hands_and_head','pos_error_upper','pos_error_lower','pos_error_pelvis']):
                    factor = 57.2958 if k in ['rot_error', 'rot_error_hands_and_head'] else 100
                    
                    final_results[steps][k] = factor * sum([results[steps][i][k] for i in range(opt['num_evaluation'])]) / opt['num_evaluation']
                
                for steps in test_times:
                    logger.log('+' * 50)
                    logger.log(f'For #steps = {steps}')
                    logger.log('Average rotational error [degree]: {:<.5f}'.format(final_results[steps]['rot_error']))
                    logger.log('Average rotational error (hands and head) [degree]: {:<.5f}'.format(final_results[steps]['rot_error_hands_and_head']))
                    # logger.log('Average rotational error (noise) [degree]: {:<.5f}'.format(final_results[steps]['noise_rot_error']))
                    logger.log('Average positional error [cm]: {:<.5f}'.format(final_results[steps]['pos_error']))
                    logger.log('Average velocity error [cm/s]: {:<.5f}'.format(final_results[steps]['vel_error']))
                    logger.log('Average positional error at hands [cm]: {:<.5f}'.format(final_results[steps]['pos_error_hands']))
                    logger.log('Average positional error at upper [cm]: {:<.5f}'.format(final_results[steps]['pos_error_upper']))
                    logger.log('Average positional error at lower [cm]: {:<.5f}'.format(final_results[steps]['pos_error_lower']))
                    logger.log('Average positional error at pelvis [cm]: {:<.5f}'.format(final_results[steps]['pos_error_pelvis']))
                    
                for j in range(start_count, max_idx):
                    print('sample ' + str(j) + ' pos_error = ' + str(sum(pos_errors_sample[j]) / len(pos_errors_sample[j]) * 100))
                    print('sample ' + str(j) + ' rot_error = ' + str(sum(rot_errors_sample[j]) / len(rot_errors_sample[j]) * 57.2958))
                    print('sample ' + str(j) + ' l_pos_error = ' + str(sum(l_pos_errors_sample[j]) / len(l_pos_errors_sample[j]) * 100))
                    print('sample ' + str(j) + ' u_pos_error = ' + str(sum(u_pos_errors_sample[j]) / len(u_pos_errors_sample[j]) * 100))
                    print('sample ' + str(j) + ' vel_error = ' + str(sum(vel_errors_sample[j]) / len(vel_errors_sample[j]) * 100))

                    file.write('sample ' + str(j) + ' pos_error = ' + str(sum(pos_errors_sample[j]) / len(pos_errors_sample[j]) * 100) + '\n')
                    file.write('sample ' + str(j) + ' rot_error = ' + str(sum(rot_errors_sample[j]) / len(rot_errors_sample[j]) * 57.2958) + '\n')
                    file.write('sample ' + str(j) + ' l_pos_error = ' + str(sum(l_pos_errors_sample[j]) / len(l_pos_errors_sample[j]) * 100) + '\n')
                    file.write('sample ' + str(j) + ' u_pos_error = ' + str(sum(u_pos_errors_sample[j]) / len(u_pos_errors_sample[j]) * 100) + '\n')
                    file.write('sample ' + str(j) + ' vel_error = ' + str(sum(vel_errors_sample[j]) / len(vel_errors_sample[j]) * 100) + '\n')

if __name__ == "__main__":
    main()