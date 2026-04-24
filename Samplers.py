import torch as th
from functorch import vmap
from tqdm import tqdm
from utils import utils_transform
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa,global2local_pose
from torch.autograd.functional import vjp

def sampling_cfg(
        diffusion,
        model,
        steps,
        joints,  # conditioning sequence [18 x T x 3]
        window_slide=20,
        ws=41,
        device='cpu',
        sampling_times=None,
    ):

    indices = list(range(steps))[::-1]
    time = joints.size(1)

    # generates noise of the shape of the sequence
    x = th.randn(6, time, 22, device=device)
    # print("X = " + str(x.shape))
    num_windows =  1 + (time - ws) // window_slide
    
    indexes = th.cat([th.arange(ws).view(1, -1)] * num_windows) + th.arange(num_windows).view(-1, 1) * window_slide
    indexes = indexes.to(device)
    # print('indexes = ' + str(indexes.size()))

    slicer = vmap(lambda idx: joints[:, idx])
    cond = slicer(indexes).to(device)
    # model.precompute_joint_embedding(cond)

    for idx, t in tqdm(enumerate(indices)):
        th.cuda.empty_cache()

        # breaks x into sliding windows
        slicer = vmap(lambda idx: x[:, idx])
        data = slicer(indexes).squeeze(1)  # slicer works

        # print('data = ' + str(data.shape))

        # filter the with the diffusion model
        t = th.tensor([t] * data.size(0), device=device)
        # print(t.shape)
        # print('t = ' + str(t[:16].size()))

        mean_mat = th.zeros(data.size(0), 6, time, 22, device=device)
        stdv_mat = th.zeros_like(mean_mat)
        ones_mat = th.zeros_like(mean_mat)
        
        model_kwargs = {'joints': cond}
        # print('cond = ' + str(cond.shape))
        
        out = diffusion.p_mean_variance(
            model, data, t, model_kwargs=model_kwargs,
            clip_denoised=True
        )

        # pred = th.permute(out['mean'], (1, 0, 2))
        # predicted_angle_global = utils_transform.sixd2matrot(E[:,:132].reshape(-1,6))
        # predicted_angle_global = predicted_angle_global.reshape(-1,22,3,3)

        for i in range(indexes.size(0)):
            mean_mat[i, :, indexes[i], :] = out['mean'][i]
            stdv_mat[i, :, indexes[i], :] = th.exp(out['log_variance'][i])
            ones_mat[i, :, indexes[i], :] = 1

        del x

        x = mean_mat.sum(dim=0) / ones_mat.sum(dim=0)
        std = stdv_mat.sum(dim=0).sqrt() / ones_mat.sum(dim=0)

        del mean_mat
        del stdv_mat
        del ones_mat
        
        if (idx - 1) != len(indices) and idx > (len(indices) - 50):
            x += std * th.zeros_like(x)
        elif (idx - 1) != len(indices) and idx <= (len(indices) - 50):
            x += std * th.randn_like(x)
            
        del std

    return x

def sampling_pigdm_batched(
        diffusion,
        model,
        steps,
        joints,  # conditioning sequence [18 x T x 3]
        A,
        y,
        grad_coeff = 5,
        window_slide=20,
        ws=41,
        device='cpu',
        sampling_times=None,
    ):

    indices = list(range(steps))[::-1]
    time = joints.size(1)

    # generates noise of the shape of the sequence
    x = th.randn(6, time, 22, device=device)
    num_windows =  1 + (time - ws) // window_slide
    
    indexes = th.cat([th.arange(ws).view(1, -1)] * num_windows) + th.arange(num_windows).view(-1, 1) * window_slide
    indexes = indexes.to(device)
    # print('indexes = ' + str(indexes.size()))

    slicer = vmap(lambda idx: joints[:, idx])
    cond = slicer(indexes).to(device)
    # model.precompute_joint_embedding(cond)

    eye = th.eye(y.shape[1], device=y.get_device())
    
    for idx, t in tqdm(enumerate(indices)):
        th.cuda.empty_cache()

        # breaks x into sliding windows
        slicer = vmap(lambda idx: x[:, idx])
        data = slicer(indexes).squeeze(1)  # slicer works

        # print('data = ' + str(data.shape))

        # filter the with the diffusion model
        t = th.tensor([t] * data.size(0), device=device)
        # print('t = ' + str(t[:16].size()))

        mean_mat = th.zeros(data.size(0), 6, time, 22, device=device)
        stdv_mat = th.zeros_like(mean_mat)
        ones_mat = th.zeros_like(mean_mat)
        grad_mat = th.zeros_like(mean_mat)
        
        step = diffusion._scale_timesteps(t[0])
        # print(step)
        noise_coeff = th.tensor(diffusion.sqrt_one_minus_alphas_cumprod[step]).to(y.get_device())
        coeff = th.tensor(diffusion.sqrt_alphas_cumprod[step]).to(y.get_device())

        data = data.requires_grad_(True)
        
            
        model_kwargs = {'joints': cond}

        def convert(model_output):
            
            # pred = th.permute(out['pred_xstart'], (0, 2, 1, 3))
            pred = th.permute(model_output, (0, 2, 1, 3))
            pred = pred.permute((0, 1, 3, 2))
            pred = pred.reshape(-1, 132)
            predicted_angle_global = utils_transform.sixd2matrot(pred[:,:132].reshape(-1,6))
            predicted_angle_global = predicted_angle_global.reshape(num_windows,-1,22,3,3)
    
            # Rotation angles in the required
            X = predicted_angle_global.swapaxes(2,3).reshape(num_windows,-1,3,66).reshape(num_windows,-1,198)
            X = X.view(num_windows,-1)

            return X

        def vectorized(diffusion, model, data, t, model_kwargs):

            data = data.reshape(-1, 6, 41, 22)
        
            out = diffusion.p_mean_variance(
                model, data, t, model_kwargs=model_kwargs,
                clip_denoised=True
            )
    
            X = convert(out["pred_xstart"])

            return out["mean"], out["variance"], X

        mean, stdv, X = vectorized(diffusion, model, data.reshape(-1, 5412), t, model_kwargs)

        inverse = th.linalg.pinv(A @ A.T + (0.1/noise_coeff.square())*eye) @ A         # PiGDM term
        # inverse = (noise_coeff.square()/0.2)*eye @ X                                     # Simple grad based term
        diff = ((y - (A @ X.detach().T).T) @ inverse)
        
        ########### Using Summing method to calculate VJP
        product = (X*diff).sum()
        grad_term = th.autograd.grad(product, data, retain_graph=True)[0].detach()
        
        ############ Using VJP directly
        # grad_term = vjp(lambda x: vectorized(diffusion, model, x.reshape(-1, 5412), t, model_kwargs)[2], data, v=diff)[1]

        for i in range(indexes.size(0)):

            mean_mat[i, :, indexes[i], :] = mean[i]
            stdv_mat[i, :, indexes[i], :] = stdv[i]
            ones_mat[i, :, indexes[i], :] = 1.
            grad_mat[i, :, indexes[i], :] = grad_term[i]

        del x

        x = mean_mat.sum(dim=0) / ones_mat.sum(dim=0)
        std = stdv_mat.sum(dim=0).sqrt() / ones_mat.sum(dim=0)
        grad = grad_mat.sum(dim=0) / ones_mat.sum(dim=0)

        del mean_mat
        del stdv_mat
        del ones_mat
        del grad_mat
        
        if (idx - 1) != len(indices) and idx > (len(indices) - 50):
            x += std * th.zeros_like(x)
        elif (idx - 1) != len(indices) and idx <= (len(indices) - 50):
            x += std * th.randn_like(x)
            
        del std

        x += grad * coeff * grad_coeff

    return x