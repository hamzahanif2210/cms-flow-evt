import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class Set2SetLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.var_transform = self.config["var_transform"]

        self.class_loss = nn.MSELoss(reduction="none")
        self.train_class = self.config.get("train_class", False)

        self.regression_loss = nn.MSELoss(reduction="none")
        self.num_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, g, scatter=True, weight=None):
        ### new code !!!
        n_pflow = g.batch_num_nodes("pflow_particles").cpu()
        n_pflow_cumsum = torch.cat([torch.tensor([0]), n_pflow.cumsum(0)])
        max_len = n_pflow.max()

        input = torch.zeros((g.batch_size, max_len, 3))
        target = torch.zeros((g.batch_size, max_len, 3))

        if self.train_class:
            input_class = torch.zeros((g.batch_size, max_len, 1))
            target_class = torch.zeros((g.batch_size, max_len, 1))
        mask = torch.zeros((g.batch_size, max_len), dtype=torch.bool)

        bs = g.batch_size
        if weight is None:
            weight = torch.ones(bs)
        weight = weight.sqrt()

        for i in range(bs):
            target[i, : n_pflow[i], 0] = (
                weight[i]
                * g.nodes["pflow_particles"].data["pt"][
                    n_pflow_cumsum[i] : n_pflow_cumsum[i + 1]
                ]
            )
            target[i, : n_pflow[i], 1] = (
                weight[i]
                * g.nodes["pflow_particles"].data["eta"][
                    n_pflow_cumsum[i] : n_pflow_cumsum[i + 1]
                ]
            )
            target[i, : n_pflow[i], 2] = (
                weight[i]
                * g.nodes["pflow_particles"].data["phi"][
                    n_pflow_cumsum[i] : n_pflow_cumsum[i + 1]
                ]
            )

            mask[i, : n_pflow[i]] = True

            input[i, : n_pflow[i], :] = (
                weight[i]
                * g.nodes["fastsim_particles"].data["pt_eta_phi_pred"][
                    n_pflow_cumsum[i] : n_pflow_cumsum[i + 1]
                ]
            )
            if self.train_class:
                input_class[i, : n_pflow[i]] = (
                    weight[i]
                    * g.nodes["fastsim_particles"].data["class_pred"][
                        n_pflow_cumsum[i] : n_pflow_cumsum[i + 1]
                    ]
                )
                target_class[i, : n_pflow[i]] = (
                    weight[i]
                    * g.nodes["pflow_particles"].data["class"][
                        n_pflow_cumsum[i] : n_pflow_cumsum[i + 1]
                    ]
                )
                # print(input[i,j,0],target[i,j,0])

        # print('input pt', input[:,:,0])
        # print('target pt', target[:,:,0])
        # new_input = input.unsqueeze(0).expand(target.size(0), -1, -1, -1)
        new_input = input.unsqueeze(1).expand(-1, target.size(1), -1, -1)

        # new_target = target.unsqueeze(1).expand(-1, input.size(0), -1, -1)
        new_target = target.unsqueeze(2).expand(-1, -1, input.size(1), -1)

        # mask_input = mask.unsqueeze(0).expand(target.size(0), -1, -1)
        mask_input = mask.unsqueeze(1).expand(-1, target.size(1), -1)

        # mask_target = mask.unsqueeze(1).expand(-1, input.size(0), -1)
        mask_target = mask.unsqueeze(2).expand(-1, -1, input.size(1))

        if self.train_class:
            new_input_class = input_class.unsqueeze(1).expand(
                -1, target.size(1), -1, -1
            )
            new_target_class = target_class.unsqueeze(2).expand(
                -1, -1, input.size(1), -1
            )
            pdist_class = (
                F.mse_loss(new_input_class, new_target_class, reduction="none").mean(-1)
                * 2
            )
            # pdist_class = -torch.log(torch.clamp(F.sigmoid(new_target_class * new_input_class), min=1e-18, max=1e18)).mean(-1) * 2
        else:
            pdist_class = 0
        pdist_pt = F.mse_loss(new_input, new_target, reduction="none")
        pdist_eta = F.mse_loss(new_input, new_target, reduction="none") * 2

        pdist_phi = (
            2
            * (
                1
                - torch.cos(
                    (new_input - new_target)
                    * g.nodes["global_node"].data["phi_std"][0].cpu()
                )
            )
            * 25
        )

        pt_mask = [x for x in range(target.size(-1)) if (x + 3) % 3 == 0]
        eta_mask = [x for x in range(target.size(-1)) if (x + 2) % 3 == 0]
        phi_mask = [x for x in range(target.size(-1)) if (x + 1) % 3 == 0]

        pdist_ptetaphi = torch.cat(
            [
                pdist_pt[:, :, :, pt_mask],
                pdist_eta[:, :, :, eta_mask],
                pdist_phi[:, :, :, phi_mask],
            ],
            dim=-1,
        )

        pdist_ptetaphi = pdist_ptetaphi.mean(
            3
        )  # pdist_ptetaphi shape (b,N,N,3) -> (b,N,N)

        ## scale class loss
        # pdist_class[new_class_target.flatten()==2] = pdist_class[new_class_target.flatten()==2] * 5

        # pdist = pdist_class + pdist_ptetaphi
        pdist = pdist_ptetaphi + pdist_class

        # set to 0 if both are fake
        mask_both0 = torch.logical_not(torch.logical_or(mask_input, mask_target))
        pdist[mask_both0] = 0

        # set to high value if only one is fake
        mask_one0 = torch.logical_xor(mask_input, mask_target)
        pdist[mask_one0] = 1e4

        pdist_ = pdist.detach().cpu().numpy()
        indices = np.array(
            [linear_sum_assignment(p) for p in pdist_]
        )  # indices shape (b,2,N)

        spicy_mat = torch.zeros((bs), device=pdist.device)
        spicy_mat_ptetaphi = torch.zeros((bs), device=pdist.device)
        spicy_mat_pt = torch.zeros((bs), device=pdist.device)
        spicy_mat_eta = torch.zeros((bs), device=pdist.device)
        spicy_mat_phi = torch.zeros((bs), device=pdist.device)
        if self.train_class:
            spicy_mat_class = torch.zeros((bs), device=pdist.device)

        for idx_i in range(pdist_ptetaphi.shape[0]):
            indices_i = indices.shape[2] * indices[idx_i, 0] + indices[idx_i, 1]

            # losses = torch.gather(
            #     pdist[idx_i].flatten(0, 1),
            #     0,
            #     torch.from_numpy(indices_i).to(device=pdist.device),
            # )
            # total_loss = losses.mean(0)

            # spicy_mat[idx_i] = total_loss

            losses = torch.gather(
                pdist_ptetaphi[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss = losses.mean(0)

            spicy_mat_ptetaphi[idx_i] = total_loss

            if self.train_class:
                losses = torch.gather(
                    pdist_class[idx_i].flatten(0, 1),
                    0,
                    torch.from_numpy(indices_i).to(device=pdist.device),
                )
                total_loss = losses.mean(0)

                spicy_mat_class[idx_i] = total_loss
            losses = torch.gather(
                pdist_class[idx_i].flatten(1, 2),
                1,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss = losses.mean(1)

            spicy_mat_class[idx_i] = total_loss

            # seperated pt eta and phi loss

            losses = torch.gather(
                pdist_pt[:, :, :, pt_mask].squeeze(3)[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            ).mean(0)
            spicy_mat_pt[idx_i] = losses

            losses = torch.gather(
                pdist_eta[:, :, :, eta_mask].squeeze(3)[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            ).mean(0)
            spicy_mat_eta[idx_i] = losses

            losses = torch.gather(
                pdist_phi[:, :, :, phi_mask].squeeze(3)[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            ).mean(0)
            spicy_mat_phi[idx_i] = losses

        # now outer Hungarian
        # indices_outer = np.array(linear_sum_assignment(spicy_mat.detach().cpu().numpy()))

        # # reordering input for plotting -->
        # i_bs_ind = indices[indices_outer[0],indices_outer[1]] # gets the matches (bs,bs,2,N) --> (bs,2,N)
        # i_bs_ind = i_bs_ind[:,1,:] # gets the input idx (bs,2,N) --> (bs,N)

        # input_reshaped_batch = input[indices_outer[1]]
        # input_reshaped_particles = input_reshaped_batch[torch.arange(i_bs_ind.shape[0]).unsqueeze(1),i_bs_ind]

        # final_input = input_reshaped_particles[:,:,-3:].flatten(0,1).detach().cpu().numpy()

        # input_reshaped_batch_class = input_class[indices_outer[1]]
        # input_reshaped_particles_class = input_reshaped_batch_class[torch.arange(i_bs_ind.shape[0]).unsqueeze(1),i_bs_ind]

        # final_input_class = np.argmax(input_reshaped_particles_class.flatten(0,1).detach().cpu().numpy(), axis=1)
        # final_input_class =  torch.multinomial(torch.nn.Softmax()(input_reshaped_particles_class.flatten(0,1)),1,replacement=True).squeeze(1).detach().cpu().numpy()
        # indices_outer_reshape = indices_outer.shape[1] * indices_outer[0] + indices_outer[1]

        # losses = torch.gather(spicy_mat_ptetaphi.flatten(), 0, torch.from_numpy(indices_outer_reshape).to(device=pdist_ptetaphi.device))
        kin_loss = spicy_mat_ptetaphi.mean(0)

        # losses = torch.gather(spicy_mat_class.flatten(), 0, torch.from_numpy(indices_outer_reshape).to(device=pdist_ptetaphi.device))
        # class_loss = losses.mean(0)

        pt_loss = spicy_mat_pt.mean(0)
        eta_loss = spicy_mat_eta.mean(0)
        phi_loss = spicy_mat_phi.mean(0)
        if self.train_class:
            class_loss = spicy_mat_class.mean(0)
        else:
            class_loss = 0

        # pt_loss = torch.gather(spicy_mat_pt.flatten(), 0, torch.from_numpy(indices_outer_reshape).to(device=pdist_ptetaphi.device)).mean(0)
        # eta_loss = torch.gather(spicy_mat_eta.flatten(), 0, torch.from_numpy(indices_outer_reshape).to(device=pdist_ptetaphi.device)).mean(0)
        # phi_loss = torch.gather(spicy_mat_phi.flatten(), 0, torch.from_numpy(indices_outer_reshape).to(device=pdist_ptetaphi.device)).mean(0)
        # final_target = target[:,:,-3:].flatten(0,1).detach().cpu().numpy()

        # print(final_input.sum(axis=1).shape)
        # mask_final = abs(final_input.sum(axis=1)) > 1e-8
        # mask_target = abs(final_target.sum(axis=1)) > 1e-8

        # mask = mask_final * mask_target

        # if self.config['per_event_scaling']:
        # pt_mean = g.nodes['global_node'].data['pt_mean'][0].cpu().numpy()
        # pt_std = g.nodes['global_node'].data['pt_std'][0].cpu().numpy()
        # eta_mean = g.nodes['global_node'].data['eta_mean'][0].cpu().numpy()
        # eta_std = g.nodes['global_node'].data['eta_std'][0].cpu().numpy()
        # phi_mean = g.nodes['global_node'].data['phi_mean'][0].cpu().numpy()
        # phi_std = g.nodes['global_node'].data['phi_std'][0].cpu().numpy()

        # else:
        #     pt_mean = self.config['var_transform']['pflow_pt']['mean']
        #     pt_std = self.config['var_transform']['pflow_pt']['std']
        #     eta_mean = self.config['var_transform']['pflow_eta']['mean']
        #     eta_std = self.config['var_transform']['pflow_eta']['std']
        #     phi_mean = self.config['var_transform']['pflow_phi']['mean']
        #     phi_std = self.config['var_transform']['pflow_phi']['std']

        # pt_fs_scaled = final_input[:,0] * pt_std + pt_mean
        # eta_fs_scaled = final_input[:,1] * eta_std + eta_mean
        # phi_fs_scaled = self.phi_reshape(final_input[:,2] * phi_std + phi_mean)

        # pt_pf_scaled = final_target[:,0] * pt_std + pt_mean
        # eta_pf_scaled = final_target[:,1] * eta_std + eta_mean
        # phi_pf_scaled = final_target[:,2] * phi_std + phi_mean

        # fs_scaled = np.stack([pt_fs_scaled,eta_fs_scaled,phi_fs_scaled],axis=1)
        # pf_scaled = np.stack([pt_pf_scaled,eta_pf_scaled,phi_pf_scaled],axis=1)
        # ptetaphi = [fs_scaled[mask], pf_scaled[mask]]
        # print(ptetaphi)

        # classes = [final_input_class[mask], target_class.flatten(0,1).detach().cpu().numpy()[mask]]
        # print(classes)

        # set size loss

        num_loss = (
            self.num_loss(
                g.nodes["global_node"].data["set_size_pred"],
                g.batch_num_nodes("pflow_particles") - 1,
            ).mean()
            / 1.5
        )
        total_loss = kin_loss + num_loss + class_loss  # + mean_loss + sigma_loss

        # if self.config['learn_class']:
        #     total_loss += class_loss

        ##### mmd+ha loss for metric tracking ---------->

        # pred_pt = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,0].reshape(g.batch_size,n_fastsim_particles)
        # pred_eta = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,1].reshape(g.batch_size,n_fastsim_particles)
        # pred_phi = g.nodes['fastsim_particles'].data['pt_eta_phi_pred'][:,2].reshape(g.batch_size,n_fastsim_particles) * self.var_transform["pflow_phi"]["std"]  + self.var_transform["pflow_phi"]["mean"]
        # targ_pt = g.nodes['pflow_particles'].data['pt'].reshape(g.batch_size,n_pflow_particles)
        # targ_eta = g.nodes['pflow_particles'].data['eta'].reshape(g.batch_size,n_pflow_particles)
        # targ_phi = g.nodes['pflow_particles'].data['phi'].reshape(g.batch_size,n_pflow_particles) * self.var_transform["pflow_phi"]["std"]  + self.var_transform["pflow_phi"]["mean"]

        # predic = torch.stack([pred_pt,pred_eta,pred_phi],axis=1)
        # target = torch.stack([targ_pt,targ_eta,targ_phi],axis=1)

        # mmd_HA_loss = self.torch_new_MMD(target,predic,HA=True)
        # mmd_PA_loss = self.torch_new_MMD(target,predic,HA=False)

        ############# <------------------

        # if scatter == True:

        # pred_set_size = np.argmax(g.nodes['global_node'].data['set_size_pred'].detach().cpu().numpy(), axis=1)
        # pred_set_size = torch.multinomial(torch.nn.Softmax()(g.nodes['global_node'].data['set_size_pred']),1,replacement=True).squeeze(1).detach().cpu().numpy()
        # set_sizes = [pred_set_size, g.batch_num_nodes('pflow_particles').detach().cpu().numpy(), g.batch_num_nodes('truth_particles').detach().cpu().numpy()]

        return {
            "total_loss": total_loss,
            "kin_loss": kin_loss.detach(),  # .detach(),
            "pt_loss": pt_loss.detach(),  # .detach(),
            "eta_loss": eta_loss.detach(),  # .detach(),
            "phi_loss": phi_loss.detach(),  # .detach(),
            "class_loss": class_loss.detach() if self.train_class else 0,
            "num_loss": num_loss.detach(),
            # 'ptetaphi': ptetaphi, 'set_size': set_sizes, #'class': classes,
            # 'mmd_HA_loss': mmd_HA_loss,
            # 'mmd_PA_loss': mmd_PA_loss,
        }

    def phi_reshape(self, inp):
        for i in range(len(inp)):
            while inp[i] > np.pi:
                inp[i] = inp[i] - 2 * np.pi
            while inp[i] < -np.pi:
                inp[i] = inp[i] + 2 * np.pi

        return inp
