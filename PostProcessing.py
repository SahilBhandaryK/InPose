import torch as th
from utils import utils_transform
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa,global2local_pose
from collections import OrderedDict

# Reconstruction with global ground truth angles
def final_prediction(E, Head_trans_global, bm, gt=False, device=None, bones=None, multiple=1.):
    # Head_trans_global = data['Head_trans_global'][:, :E.shape[0]].squeeze().to(device)
    # E = E['sample'].reshape(-1, 132)
    # E = data['H'].to(device)
    if not gt:
        E = E.permute((0, 2, 1))
    E = E.reshape(-1, 132)
    
    predicted_angle_global = utils_transform.sixd2matrot(E[:,:132].reshape(-1,6).detach())
    predicted_angle_global = predicted_angle_global.reshape(-1,22,3,3)
    predicted_angle = global2local_pose(predicted_angle_global, bm.kintree_table[0][:22].long())
    predicted_angle = matrot2aa(predicted_angle.reshape(-1,3,3)).reshape(E[:,:132].shape[0],-1).float()

    # Calculate global translation

    T_head2world = Head_trans_global.clone()
    T_head2root_pred = th.eye(4).repeat(T_head2world.shape[0],1,1).cuda()
    rotation_local_matrot = aa2matrot(th.cat([th.zeros([predicted_angle.shape[0],3]).cuda(),predicted_angle[...,3:66]],dim=1).reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
    rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0][:22].long())
    head2root_rotation = rotation_global_matrot[:,15,:]

    body_pose_local_pred=bm(**{'pose_body':predicted_angle[...,3:66],'bones':bones})
    head2root_translation = body_pose_local_pred.Jtr[:,15,:] * multiple
    T_head2root_pred[:,:3,:3] = head2root_rotation
    T_head2root_pred[:,:3,3] = head2root_translation
    t_head2world = T_head2world[:,:3,3].clone() * multiple
    T_head2world[:,:3,3] = 0
    T_root2world_pred = th.matmul(T_head2world, th.inverse(T_head2root_pred))

    rotation_root2world_pred = matrot2aa(T_root2world_pred[:,:3,:3])
    translation_root2world_pred = T_root2world_pred[:,:3,3]
    body_pose_local=bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'bones':bones})
    position_global_full_local = body_pose_local.Jtr[:,:22,:] * multiple
    t_head2root = position_global_full_local[:,15,:]
    t_root2world = (-t_head2root+t_head2world.cuda()) / multiple

    predicted_body=bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world, 'bones':bones}) 
    # No stabilizer: 'root_orient':rotation_root2world_pred.cuda()

    predicted_position = predicted_body.Jtr[:,:22,:] * multiple
    
    predicted_translation = t_root2world

    body_parms = OrderedDict()
    body_parms['pose_body'] = predicted_angle[...,3:66]
    body_parms['vrot'] = predicted_angle_global
    body_parms['root_orient'] = predicted_angle[...,:3]
    body_parms['trans'] = predicted_translation
    body_parms['position'] = predicted_position
    body_parms['position_notran'] = position_global_full_local
    body_parms['body'] = predicted_body

    return body_parms
          
def compute_mean_and_cov(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    
# Reconstruction with local ground truth angles
def final_prediction_local(E, Head_trans_global, bm,gt=False, device=None, j_noise=0., bones=None, multiple=1.):
    
    # Head_trans_global = data['Head_trans_global'][:, :E.shape[0]].squeeze().to(device)
    # E = E['sample'].reshape(-1, 132)
    # E = data['H'].to(device)
    if not gt:
        E = E.permute((0, 2, 1))
    E = E.reshape(-1, 132)
    
    predicted_angle = utils_transform.sixd2aa(E[:,:132].reshape(-1,6).detach())
    
    predicted_angle_local = utils_transform.sixd2matrot(E[:,:132].reshape(-1,6).detach())
    predicted_angle_local = predicted_angle_local.reshape(-1,22,3,3)
    predicted_angle_global = local2global_pose(predicted_angle_local, bm.kintree_table[0][:22].long())

    predicted_angle = predicted_angle.reshape(E[:,:132].shape[0],-1).float()
        
    # Calculate global translation
    
    T_head2world = Head_trans_global.clone()
    T_head2root_pred = th.eye(4).repeat(T_head2world.shape[0],1,1).cuda()
    
    rotation_local_matrot = aa2matrot(th.cat([th.zeros([predicted_angle.shape[0],3]).cuda(),predicted_angle[...,3:66]],dim=1).reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
    rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0][:22].long())
    # print('theirs = ' + str(rotation_global_matrot.shape))
    head2root_rotation = rotation_global_matrot[:,15,:]

    body_pose_local_pred=bm(**{'pose_body':predicted_angle[...,3:66], 'j_noise':j_noise,'bones':bones})
    head2root_translation = body_pose_local_pred.Jtr[:,15,:] * multiple
    T_head2root_pred[:,:3,:3] = head2root_rotation
    T_head2root_pred[:,:3,3] = head2root_translation
    t_head2world = T_head2world[:,:3,3].clone() * multiple
    T_head2world[:,:3,3] = 0
    T_root2world_pred = th.matmul(T_head2world, th.inverse(T_head2root_pred))

    rotation_root2world_pred = matrot2aa(T_root2world_pred[:,:3,:3])
    translation_root2world_pred = T_root2world_pred[:,:3,3]
    body_pose_local=bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'j_noise':j_noise,'bones':bones})
    position_global_full_local = body_pose_local.Jtr[:,:22,:] *multiple
    t_head2root = position_global_full_local[:,15,:]
    t_root2world = (-t_head2root+t_head2world.cuda()) / multiple

    predicted_body=bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world, 'j_noise':j_noise,'bones':bones}) 
    # No stabilizer: 'root_orient':rotation_root2world_pred.cuda()

    predicted_position = predicted_body.Jtr[:,:22,:] * multiple
    
    predicted_translation = t_root2world

    rotation_global_matrot[:,0,:,:] = T_root2world_pred[:,:3,:3]

    body_parms = OrderedDict()
    body_parms['pose_body'] = predicted_angle[...,3:66]
    body_parms['vrot'] = predicted_angle_global
    body_parms['root_orient'] = predicted_angle[...,:3]
    body_parms['trans'] = predicted_translation
    body_parms['position'] = predicted_position
    body_parms['position_notran'] = position_global_full_local
    body_parms['body'] = predicted_body

    return body_parms