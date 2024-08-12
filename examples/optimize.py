# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh
import os
from util import *
import render
import loss
import imageio

import sys
sys.path.append('..')
from flexicubes import FlexiCubes

###############################################################################
# Functions adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

def lr_schedule(iter):
    # print(iter, max(0.0, 10**(-(iter)*0.0002)))
    # return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    
    # return max(0.0, (10**(-1-(iter)*0.0002))) # Exponential falloff from [0.1, 0.01] over 5k epochs.    
    return 0.01

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='flexicubes optimization')
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)    
    
    parser.add_argument('-i', '--iter', type=int, default=4000)
    parser.add_argument('-b', '--batch', type=int, default=4)
    parser.add_argument('-r', '--train_res', nargs=2, type=int, default=[256, 256])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--voxel_grid_res', type=int, default=80)
    
    parser.add_argument('--sdf_loss', type=bool, default=False)
    parser.add_argument('--develop_reg', type=bool, default=False)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)
    
    parser.add_argument('-dr', '--display_res', nargs=2, type=int, default=[256, 256])
    parser.add_argument('-si', '--save_interval', type=int, default=20)
    FLAGS = parser.parse_args()
    device = 'cuda'
    
    render_mode = False

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    glctx = dr.RasterizeGLContext()

    np.random.seed(0)
    
    # Load GT mesh
    print(f"Loading reference mesh from {FLAGS.ref_mesh}")
    gt_mesh = load_mesh(FLAGS.ref_mesh, device)
    gt_mesh.auto_normals() # compute face normals for visualization
    
    # ==============================================================================================
    #  Create and initialize FlexiCubes
    # ==============================================================================================
    fc = FlexiCubes(device)
    x_nx3, cube_fx8 = fc.construct_voxel_grid(FLAGS.voxel_grid_res)
    x_nx3 *= 2 # scale up the grid so that it's larger than the target object

    import mesh2sdf
    sdf, _ = mesh2sdf.compute(
    gt_mesh.vertices.detach().cpu().numpy(), gt_mesh.faces.detach().cpu().numpy(), FLAGS.voxel_grid_res+1, fix=True, level=2/(FLAGS.voxel_grid_res+1), return_mesh=True)
    print(sdf.shape)
    
    # sdf = torch.rand_like(x_nx3[:,0]) - 0.1 # randomly init SDF
    # print(sdf.shape)
    sdf = torch.tensor(sdf.flatten(), dtype=torch.float32, device=device)
    sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
    # set per-cube learnable weights to zeros
    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
    weight    = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)
    
    #  Retrieve all the edges of the voxel grid; these edges will be utilized to 
    #  compute the regularization loss in subsequent steps of the process.    
    all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
    grid_edges = torch.unique(all_edges, dim=0)
    
    # ==============================================================================================
    #  Setup optimizer
    # ==============================================================================================
    # trainable_normalmap = torch.tensor(torch.zeros((FLAGS.train_res[0], FLAGS.train_res[1], 3)), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([sdf, weight,deform], lr=FLAGS.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x)) 

    from PIL import Image
    from pathlib import Path
    mesh_path = Path(FLAGS.ref_mesh)
    # tex_path = mesh_path.parent / "texture_0.png"
    tex_path = mesh_path.parent / "tex_base_color.jpg"
    tex = Image.open(tex_path)
    tex = np.array(tex)
    tex = tex / 255
    tex = tex[::-1, :]
    tex = torch.tensor(tex.copy(), dtype=torch.float32, device=device)


    # ==============================================================================================
    #  Train loop
    # ==============================================================================================   
    for it in range(FLAGS.iter): 
        optimizer.zero_grad()
        # sample random camera poses
        mv, mvp = render.get_random_camera_batch(FLAGS.batch, iter_res=FLAGS.train_res, device=device, use_kaolin=False)
        # render gt mesh
        target = render.render_mesh_paper(gt_mesh, mv, mvp, FLAGS.train_res, return_types=["mask", "depth", "alpha"])
        # extract and render FlexiCubes mesh
        grid_verts = x_nx3 + (2-1e-8) / (FLAGS.voxel_grid_res * 2) * torch.tanh(deform)
        vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
            gamma_f=weight[:,20], training=True)
        # print(vertices.detach().cpu().numpy().max(), vertices.detach().cpu().numpy().min())
        flexicubes_mesh = Mesh(vertices, faces)
        flexicubes_mesh.auto_normals()
        buffers = render.render_mesh_paper(flexicubes_mesh, mv, mvp, FLAGS.train_res, return_types=["mask", "depth", "alpha"])
        # print(mv.shape, mvp.shape)
        rendered_cur_buffers = render.render_mesh_paper(flexicubes_mesh, mv[0].unsqueeze(0), mvp[0].unsqueeze(0), FLAGS.train_res, return_types=["vertex_normal", "normal"])
        rendered_gt_buffers = render.render_mesh_paper(gt_mesh, mv[0].unsqueeze(0), mvp[0].unsqueeze(0), FLAGS.train_res, return_types=["vertex_normal", "normal", "color"], tex=tex)

        # evaluate reconstruction loss
        mask_loss = (buffers['mask'] - target['mask']).abs().mean()
        # mask_loss = 0
        # depth_loss = (((((buffers['depth'] - (target['depth']))* target['mask'])**2).sum(-1)+1e-8)).sqrt().mean() * 10
        depth_loss = 0

        if render_mode:
            # Pre rendering color and normal
            targe_normal = ((rendered_gt_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
            targe_color = ((rendered_gt_buffers["color"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
            imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}_normal.png'.format(it)), np.concatenate([targe_color], 1))
            normal_loss = 0
        else:
            # targe_normal_rgb = Image.open("../../intrinsic-lora/output/normal/predicted/{:04d}_normal_normal.png".format(it))
            targe_normal_rgb = Image.open("out/head_normal/{:04d}_normal_normal.png".format(it))
            # targe_normal_rgb = Image.open("out/head_normal/0000_normal_normal.png".format(it))
            targe_normal_rgb = targe_normal_rgb.resize((FLAGS.train_res[0], FLAGS.train_res[1]))
            targe_normal_rgb = np.array(targe_normal_rgb)
            targe_normal = targe_normal_rgb / 255
            targe_normal = torch.tensor(targe_normal).to(device)
            targe_normal *= target["mask"][0]
            targe_normal_rgb = (targe_normal.detach().cpu().numpy()*255).astype(np.uint8)

            current_normal = (rendered_cur_buffers["normal"][0]+1)/2
            # current_normal =  trainable_normalmap
            # print(current_normal.shape)
            # # normal_loss = (((current_normal - targe_normal)*target['mask'][0])**2).sum(-1).sqrt().mean() * 0.01
            # normal_loss = ((((current_normal* target['mask'][0] - (targe_normal* target['mask'][0]))**2).sum(-1)+1e-8)).sqrt().mean() * 0.1
            import torch.nn.functional as F
            normal_loss = F.mse_loss((current_normal*target['mask'][0]).float(), (targe_normal*target['mask'][0]).float())
            # print(normal_loss.dtype)
            # normal_loss = 0

            if it % 50 == 0:
                # current_normal_rgb = (((rendered_cur_buffers["normal"][0].detach().cpu().numpy()+1)/2)*target["mask"][0].detach().cpu().numpy()*255).astype(np.uint8)
                current_normal_rgb = ((current_normal*target["mask"][0]).detach().cpu().numpy()*255).astype(np.uint8)
                imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}_normal.png'.format(it)), np.concatenate([current_normal_rgb, targe_normal_rgb], 1))


        t_iter = it / FLAGS.iter
        sdf_weight = FLAGS.sdf_regularizer - (FLAGS.sdf_regularizer - FLAGS.sdf_regularizer/20)*min(1.0, 4.0 * t_iter)
        reg_loss = loss.sdf_reg_loss(sdf, grid_edges).mean() * sdf_weight # Loss to eliminate internal floaters that are not visible
        reg_loss += L_dev.mean() * 0.5
        reg_loss += (weight[:,:20]).abs().mean() * 0.1
        total_loss = mask_loss + depth_loss + reg_loss + normal_loss
        # total_loss = torch.nn.functional.mse_loss(torch.tensor([0.0]), torch.tensor([0.0]))
        # total_loss.requires_grad = True
        
        if FLAGS.sdf_loss: # optionally add SDF loss to eliminate internal structures
            with torch.no_grad():
                pts = sample_random_points(1000, gt_mesh)
                gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
            pred_sdf = compute_sdf(pts, flexicubes_mesh.vertices, flexicubes_mesh.faces)
            total_loss += torch.nn.functional.mse_loss(pred_sdf, gt_sdf) * 2e3
        
        # optionally add developability regularizer, as described in paper section 5.2
        if FLAGS.develop_reg:
            reg_weight = max(0, t_iter - 0.8) * 5
            if reg_weight > 0: # only applied after shape converges
                reg_loss = loss.mesh_developable_reg(flexicubes_mesh).mean() * 10
                reg_loss += (deform).abs().mean()
                reg_loss += (weight[:,:20]).abs().mean()
                total_loss = mask_loss + depth_loss + reg_loss 
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()        
        
        if (it % FLAGS.save_interval == 0 or it == (FLAGS.iter-1)): # save normal image for visualization
            with torch.no_grad():
                # extract mesh with training=False
                vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, FLAGS.voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20],
                gamma_f=weight[:,20], training=False)
                flexicubes_mesh = Mesh(vertices, faces)
                
                flexicubes_mesh.auto_normals() # compute face normals for visualization
                mv, mvp = render.get_rotate_camera(it//FLAGS.save_interval, iter_res=FLAGS.display_res, device=device,use_kaolin=False)
                # print(mv.shape, mvp.shape)
                val_buffers = render.render_mesh_paper(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                val_image = ((val_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                
                gt_buffers = render.render_mesh_paper(gt_mesh, mv.unsqueeze(0), mvp.unsqueeze(0), FLAGS.display_res, return_types=["normal"], white_bg=True)
                gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy()+1)/2*255).astype(np.uint8)
                # imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}.png'.format(it)), np.concatenate([val_image, gt_image], 1))
                if render_mode:
                    print(f"Optimization Step [{it}/{FLAGS.iter}], Loss: {total_loss.item():.4f}, Mask Loss: {mask_loss.item():.4f}")
                else:
                    print(f"Optimization Step [{it}/{FLAGS.iter}], Loss: {total_loss.item():.4f}, Normal Loss: {normal_loss.item():.4f}, Mask Loss: {mask_loss.item():.4f}")

                mask = (buffers["mask"][0]*255).detach().cpu().numpy().astype(np.uint8)
                # print(mask.shape)
                # imageio.imwrite(os.path.join(FLAGS.out_dir, '{:04d}_mask.jpg'.format(it)), mask)
                # import cv2
                # cv2.imwrite(os.path.join(FLAGS.out_dir, '{:04d}_mask.jpg'.format(it)), mask)

            
    # ==============================================================================================
    #  Save ouput
    # ==============================================================================================     
    mesh_np = trimesh.Trimesh(vertices = vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh_np.export(os.path.join(FLAGS.out_dir, 'output_mesh.obj'))