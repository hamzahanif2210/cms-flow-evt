import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import uproot as uproot
from scipy.optimize import linear_sum_assignment
from scipy.stats import iqr
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import energyflow as ef


def reshape_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


fs = 18
c_tr = "black"
c_fs = "crimson"
c_pf = "#1f77b4"


def matching(
    input_data, truth_data, class_match=False, dr_match=False, mask_fn=None, dr_cut=0.6
):
    def mse(a, b):
        return (a - b) ** 2

    input_pt = input_data["pt"]
    input_eta = input_data["eta"]
    input_phi = input_data["phi"]

    truth_pt = truth_data["pt"]
    truth_eta = truth_data["eta"]
    truth_phi = truth_data["phi"]

    input_jet_idx = input_data["idx"]
    truth_jet_idx = truth_data["idx"]

    if class_match:
        print("Using class matching")
        input_class = input_data["class"]
        truth_class = truth_data["class"]
    else:
        input_class = None
        truth_class = None

    input_indices = []
    truth_indices = []
    hung_cost = np.zeros(len(input_pt))

    for i in tqdm(range(len(input_pt)), desc="matching"):
        # for i in range(1):
        if mask_fn is not None:
            mask_input = mask_fn(input_jet_idx[i])
            mask_truth = mask_fn(truth_jet_idx[i])
        else:
            mask_input = np.ones_like(input_jet_idx[i]) > 0
            mask_truth = np.ones_like(truth_jet_idx[i]) > 0

        input_pt_i = np.log(
            input_pt[i][mask_input] * 1000, where=input_pt[i][mask_input] > 0
        )
        input_eta_i = input_eta[i][mask_input]
        input_phi_i = input_phi[i][mask_input]

        truth_pt_i = np.log(
            truth_pt[i][mask_truth] * 1000, where=truth_pt[i][mask_truth] > 0
        )
        truth_eta_i = truth_eta[i][mask_truth]
        truth_phi_i = truth_phi[i][mask_truth]

        if input_class is not None:
            input_class_i = input_class[i][mask_input] < 3
            truth_class_i = truth_class[i][mask_truth] < 3

            truth_class_i = np.tile(
                np.expand_dims(truth_class_i, axis=1), (1, len(input_class_i))
            )  # row content same
            input_class_i = np.tile(
                np.expand_dims(input_class_i, axis=0), (len(truth_class_i), 1)
            )

        truth_pt_i = np.tile(
            np.expand_dims(truth_pt_i, axis=1), (1, len(input_pt_i))
        )  # row content same
        input_pt_i = np.tile(
            np.expand_dims(input_pt_i, axis=0), (len(truth_pt_i), 1)
        )  # column content same

        truth_eta_i = np.tile(
            np.expand_dims(truth_eta_i, axis=1), (1, len(input_eta_i))
        )  # row content same
        input_eta_i = np.tile(
            np.expand_dims(input_eta_i, axis=0), (len(truth_eta_i), 1)
        )  # column content same

        truth_phi_i = np.tile(
            np.expand_dims(truth_phi_i, axis=1), (1, len(input_phi_i))
        )  # row content same
        input_phi_i = np.tile(
            np.expand_dims(input_phi_i, axis=0), (len(truth_phi_i), 1)
        )  # column content same

        if dr_match:
            loss_pt = 0
            # loss_phi = mse(truth_phi_i, input_phi_i)
            # loss_eta = mse(truth_eta_i, input_eta_i)
            loss_phi = np.pow(reshape_phi(truth_phi_i - input_phi_i), 2)
            loss_eta = np.power(truth_eta_i - input_eta_i, 2)
        else:
            loss_pt = mse(truth_pt_i, input_pt_i) * 0.1
            # loss_eta = mse(truth_eta_i, input_eta_i)  # * 10
            # # loss_phi = 2 * (1 - np.cos(truth_phi_i - input_phi_i)) * 10
            # loss_phi = mse(truth_phi_i, input_phi_i)
            loss_phi = np.pow(reshape_phi(truth_phi_i - input_phi_i), 2)
            loss_eta = np.power(truth_eta_i - input_eta_i, 2)

        if input_class is not None:
            loss_class = torch.nn.BCELoss(reduction="none")(
                torch.tensor(input_class_i).float(), torch.tensor(truth_class_i).float()
            ).numpy()

        # print(loss_pt.mean())
        # print(loss_eta.mean())
        # print(loss_phi.mean())
        # print('------------------')

        if np.isnan(loss_pt).all():
            loss = loss_eta + loss_phi

        else:
            loss = loss_pt + loss_eta + loss_phi
        loss_hung = loss.copy()

        if input_class is not None:
            loss += loss_class

        dr = np.sqrt(loss_eta + loss_phi)
        loss[dr > dr_cut] = 1e3

        loss[loss == np.inf] = 1000
        loss[loss == np.nan] = 1000
        truth_ix, input_ix = linear_sum_assignment(loss)

        # Create boolean mask for dr < 0.6
        mask = dr <= dr_cut

        # Find all pairs (i, j) where dr < 0.6
        filtered_pairs = np.argwhere(mask)

        # Convert truth_ix and input_ix into sets for fast lookups
        truth_input_pairs = set(zip(truth_ix, input_ix))

        # Select new pairs where (i, j) are in the filtered_pairs and truth_input_pairs
        new_pairs = np.array(
            [pair for pair in filtered_pairs if tuple(pair) in truth_input_pairs]
        )

        # Split new pairs into new_truth_ix and new_input_ix
        if len(new_pairs) == 0:
            new_truth_ix = []
            new_input_ix = []
        else:
            new_truth_ix, new_input_ix = new_pairs.T

        input_ix = np.array(new_input_ix)
        truth_ix = np.array(new_truth_ix)

        if len(new_pairs) == 0:
            hung_cost[i] = 1000
            # print(i)
        else:
            indices = input_ix.shape[0] * truth_ix + input_ix
            loss_extract = np.take_along_axis(loss_hung.flatten(), indices, axis=0)
            hung_cost[i] = loss_extract.mean()

        input_indices.append(input_ix)
        truth_indices.append(truth_ix)

    print("Matching done!")

    return truth_indices, input_indices, hung_cost


def matching_dr(input_data, truth_data, mask_fn=None, dr_cut=0.6):
    def mse(a, b):
        return (a - b) ** 2

    input_eta = input_data["eta"]
    input_phi = input_data["phi"]

    truth_eta = truth_data["eta"]
    truth_phi = truth_data["phi"]

    input_jet_idx = input_data["idx"]
    truth_jet_idx = truth_data["idx"]

    input_indices = []
    truth_indices = []
    hung_cost = np.zeros(len(input_eta))

    for i in tqdm(range(len(input_eta)), desc="matching"):
        # for i in range(1):
        if mask_fn is not None:
            mask_input = mask_fn(input_jet_idx[i])
            mask_truth = mask_fn(truth_jet_idx[i])
        else:
            mask_input = np.ones_like(input_jet_idx[i]) > 0
            mask_truth = np.ones_like(truth_jet_idx[i]) > 0

        input_eta_i = input_eta[i][mask_input]
        input_phi_i = input_phi[i][mask_input]

        truth_eta_i = truth_eta[i][mask_truth]
        truth_phi_i = truth_phi[i][mask_truth]

        truth_eta_i = np.tile(
            np.expand_dims(truth_eta_i, axis=1), (1, len(input_eta_i))
        )  # row content same
        input_eta_i = np.tile(
            np.expand_dims(input_eta_i, axis=0), (len(truth_eta_i), 1)
        )  # column content same

        truth_phi_i = np.tile(
            np.expand_dims(truth_phi_i, axis=1), (1, len(input_phi_i))
        )  # row content same
        input_phi_i = np.tile(
            np.expand_dims(input_phi_i, axis=0), (len(truth_phi_i), 1)
        )  # column content same

        loss_phi = mse(truth_phi_i, input_phi_i)
        loss_eta = mse(truth_eta_i, input_eta_i)

        loss = loss_eta + loss_phi

        loss_hung = loss.copy()

        dr = np.sqrt(loss_eta + loss_phi)
        loss[dr > dr_cut] = 1e3

        loss[loss == np.inf] = 1000
        loss[loss == np.nan] = 1000
        truth_ix, input_ix = linear_sum_assignment(loss)

        # Create boolean mask for dr < 0.6
        mask = dr <= dr_cut

        # Find all pairs (i, j) where dr < 0.6
        filtered_pairs = np.argwhere(mask)

        # Convert truth_ix and input_ix into sets for fast lookups
        truth_input_pairs = set(zip(truth_ix, input_ix))

        # Select new pairs where (i, j) are in the filtered_pairs and truth_input_pairs
        new_pairs = np.array(
            [pair for pair in filtered_pairs if tuple(pair) in truth_input_pairs]
        )

        # Split new pairs into new_truth_ix and new_input_ix
        if len(new_pairs) == 0:
            new_truth_ix = []
            new_input_ix = []
        else:
            new_truth_ix, new_input_ix = new_pairs.T

        input_ix = np.array(new_input_ix)
        truth_ix = np.array(new_truth_ix)

        if len(new_pairs) == 0:
            hung_cost[i] = 1000
            # print(i)
        else:
            indices = input_ix.shape[0] * truth_ix + input_ix
            loss_extract = np.take_along_axis(loss_hung.flatten(), indices, axis=0)
            hung_cost[i] = loss_extract.mean()

        input_indices.append(input_ix)
        truth_indices.append(truth_ix)

    print("Matching done!")

    return truth_indices, input_indices, hung_cost


def plot_1d(
    pf_data,
    fs_data,
    tr_data,
    plot_path,
    save_fig=True,
    mask_fn=None,
    fig_kwargs={},
):
    print("Make 1d plots ... ")

    bins = {
        "pt": np.linspace(0, 200, 60),
        "eta": np.linspace(-3, 3, 60),
        "phi": np.linspace(-np.pi, np.pi, 60),
        "vx": np.linspace(-2, 2, 60),
        "vy": np.linspace(-2, 2, 60),
        "vz": np.linspace(-15, 15, 60),
    }

    fig, ax = plt.subplots(2, 3, figsize=(20, 8), dpi=100, tight_layout=True)
    ax = ax.flatten()
    if mask_fn is not None:
        tr_mask = mask_fn(np.concatenate(tr_data["idx"]))
        pf_mask = mask_fn(np.concatenate(pf_data["idx"]))
        fs_mask = mask_fn(np.concatenate(fs_data["idx"]))

    for i, key in enumerate(bins.keys()):
        pf_data_i = np.concatenate(pf_data[key])
        fs_data_i = np.concatenate(fs_data[key])
        tr_data_i = np.concatenate(tr_data[key])
        if mask_fn is not None:
            pf_data_i = pf_data_i[pf_mask]
            fs_data_i = fs_data_i[fs_mask]
            tr_data_i = tr_data_i[tr_mask]
        ax[i].hist(
            pf_data_i,
            bins=bins[key],
            histtype="stepfilled",
            color=c_pf,
            label="pflow",
            alpha=0.4,
            density=True,
        )
        ax[i].hist(
            fs_data_i,
            bins=bins[key],
            histtype="step",
            color=c_fs,
            label="fastsim",
            density=True,
        )
        ax[i].hist(
            tr_data_i,
            bins=bins[key],
            histtype="step",
            color=c_tr,
            label="truth",
            density=True,
        )
        ax[i].set_xlabel(key, fontsize=fs)
        ax[i].set_ylabel("normalized events", fontsize=fs)
        ax[i].tick_params(labelsize=fs)
        ax[i].legend(fontsize=fs, frameon=False)
        if key in ["pt", "vx", "vy"]:
            ax[i].set_yscale("log")
    if save_fig:
        fig.savefig(plot_path + "1d_plot.png", format="png")
        plt.close(fig)
    print("1d plot saved")


def cardinality_plot(
    pf_data,
    fs_data,
    tr_data,
    plot_path,
    save_fig=True,
    mask_fn=None,
    bins=np.linspace(0, 300, 100),
    res_bins=np.linspace(-150, 200, 50),
):
    tr_card = np.zeros(len(pf_data["pt"]))
    pf_card = np.zeros(len(pf_data["pt"]))
    fs_card = np.zeros(len(pf_data["pt"]))

    for i in tqdm(range(len(tr_card)), desc="cardinality"):
        if mask_fn is not None:
            n_tr = np.sum(mask_fn(tr_data["idx"][i]))
            n_pf = np.sum(mask_fn(pf_data["idx"][i]))
            n_fs = np.sum(mask_fn(fs_data["idx"][i]))
        else:
            n_tr = len(tr_data["pt"][i])
            n_pf = len(pf_data["pt"][i])
            n_fs = len(fs_data["pt"][i])

        tr_card[i] = n_tr
        pf_card[i] = n_pf
        fs_card[i] = n_fs

    # plot overall residual cardinality plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=100, tight_layout=True)

    ax[0].hist(tr_card, bins=bins, histtype="step", color=c_tr, label="truth")
    ax[0].hist(
        pf_card, bins=bins, histtype="stepfilled", color=c_pf, label="pflow", alpha=0.4
    )
    ax[0].hist(fs_card, bins=bins, histtype="step", color=c_fs, label="fastsim")
    ax[0].set_xlabel("Cardinality", fontsize=fs)
    ax[0].set_ylabel("Events", fontsize=fs)
    ax[0].tick_params(labelsize=fs)
    ax[0].legend(fontsize=fs, frameon=False)

    ax[1].hist(
        tr_card - pf_card,
        bins=res_bins,
        histtype="stepfilled",
        color=c_pf,
        label="pflow",
        alpha=0.4,
    )
    ax[1].hist(
        tr_card - fs_card, bins=res_bins, histtype="step", color=c_fs, label="fastsim"
    )
    ax[1].set_xlabel("Truth - reco cardinality", fontsize=fs)
    ax[1].set_ylabel("Events", fontsize=fs)
    ax[1].tick_params(labelsize=fs)
    ax[1].legend(fontsize=fs, frameon=False)

    fig.tight_layout()
    if save_fig:
        fig.savefig(plot_path + "cardinality.png", format="png")
        plt.close(fig)
    else:
        fig.show()


def calc_residuals(tr_indices, indices, data, tr_data):
    varlist = ["pt", "eta", "phi", "vx", "vy", "vz"]
    res_pf = {}, {}
    n_pf = 0, 0
    for i in range(len(indices)):
        if len(indices[i]) == 0:
            continue
        n_pf += len(data["pt"][i][indices[i]])

    for var in varlist:
        res_pf[var] = np.zeros(n_pf)

    curr_pf_len = 0

    for i in tqdm(range(len(indices)), desc="residuals"):
        if len(indices[i]) == 0:
            continue
        for var in varlist:
            data_ = data[var][i][indices[i]]
            tr_data_ = tr_data[var][i]

            if len(data_) == 0 or len(tr_data_[tr_indices[i]]) == 0:
                continue
            res_pf[var][curr_pf_len : curr_pf_len + len(data_)] = (
                tr_data_[tr_indices[i]] - data_
            )
            if var == "pt":
                res_pf[var][curr_pf_len : curr_pf_len + len(data_)] /= tr_data_[
                    tr_indices[i]
                ]

        curr_pf_len += len(data_)
    for var in varlist:
        res_pf[var] = res_pf[var][:curr_pf_len]
    res_pf["phi"] = reshape_phi(res_pf["phi"])

    return res_pf


def residuals_1d(
    tr_pf_indices,
    pf_indices,
    tr_fs_indices,
    fs_indices,
    pf_data,
    fs_data,
    tr_data,
    plot_path,
    save_fig=True,
    mask_fn=None,
):
    varlist = ["pt", "eta", "phi", "vx", "vy", "vz"]
    res_pf, res_fs = {}, {}
    n_pf, n_fs = 0, 0
    for i in range(len(pf_indices)):
        if len(pf_indices[i]) == 0 or len(fs_indices[i]) == 0:
            continue
        if mask_fn is not None:
            n_pf += len(pf_data["pt"][i][mask_fn(pf_data["idx"][i])][pf_indices[i]])
            n_fs += len(fs_data["pt"][i][mask_fn(fs_data["idx"][i])][fs_indices[i]])
        else:
            n_pf += len(pf_data["pt"][i][pf_indices[i]])
            n_fs += len(fs_data["pt"][i][fs_indices[i]])
    for var in varlist:
        res_pf[var] = np.zeros(n_pf)
        res_fs[var] = np.zeros(n_fs)

    curr_pf_len = 0
    curr_fs_len = 0

    for i in tqdm(range(len(pf_indices)), desc="residuals"):
        if len(pf_indices[i]) == 0 or len(fs_indices[i]) == 0:
            continue
        if mask_fn is not None:
            mask_pf = mask_fn(pf_data["idx"][i])
            mask_fs = mask_fn(fs_data["idx"][i])
            mask_tr = mask_fn(tr_data["idx"][i])
        else:
            mask_pf = np.ones_like(pf_data["pt"][i]) > 0
            mask_fs = np.ones_like(fs_data["pt"][i]) > 0
            mask_tr = np.ones_like(tr_data["pt"][i]) > 0
        # if np.sum(mask_pf) == 0 or np.sum(mask_fs) == 0 or np.sum(mask_tr) == 0:
        #     continue

        # if len(tr_pf_indices[i]) == 0 or len(tr_fs_indices[i]) == 0:
        #     continue

        for var in varlist:
            pf_data_ = pf_data[var][i][mask_pf][pf_indices[i]]
            fs_data_ = fs_data[var][i][mask_fs][fs_indices[i]]
            tr_data_ = tr_data[var][i][mask_tr]

            if (
                len(pf_data_) == 0
                or len(fs_data_) == 0
                or len(tr_data_[tr_pf_indices[i]]) == 0
            ):
                continue
            res_pf[var][curr_pf_len : curr_pf_len + len(pf_data_)] = (
                tr_data_[tr_pf_indices[i]] - pf_data_
            )
            res_fs[var][curr_fs_len : curr_fs_len + len(fs_data_)] = (
                tr_data_[tr_fs_indices[i]] - fs_data_
            )
            if var == "pt":
                res_pf[var][curr_pf_len : curr_pf_len + len(pf_data_)] /= tr_data_[
                    tr_pf_indices[i]
                ]
                res_fs[var][curr_fs_len : curr_fs_len + len(fs_data_)] /= tr_data_[
                    tr_fs_indices[i]
                ]

        curr_pf_len += len(pf_data_)
        curr_fs_len += len(fs_data_)
    for var in varlist:
        res_pf[var] = res_pf[var][:curr_pf_len]
        res_fs[var] = res_fs[var][:curr_fs_len]
    res_pf["phi"] = reshape_phi(res_pf["phi"])
    res_fs["phi"] = reshape_phi(res_fs["phi"])

    fig, ax = plt.subplots(2, 3, figsize=(20, 8), dpi=100, tight_layout=True)
    ax = ax.flatten()
    bins = np.linspace(-1, 1, 50)
    # bins_eta = np.linspace(min(np.min(res_eta_fs),np.min(res_eta_pf)),max(np.max(res_eta_fs),np.max(res_eta_pf)),50)
    # bins_phi = np.linspace(min(np.min(res_phi_fs),np.min(res_phi_pf)),max(np.max(res_phi_fs),np.max(res_phi_pf)),50)
    for i, var in enumerate(varlist):
        ax[i].hist(
            res_pf[var],
            bins=bins,
            histtype="stepfilled",
            color=c_pf,
            label="pflow",
            alpha=0.4,
            density=True,
        )
        ax[i].hist(
            res_fs[var],
            bins=bins,
            histtype="step",
            color=c_fs,
            label="fastsim",
            density=True,
        )
        ax[i].set_xlabel(var, fontsize=fs)
        ax[i].set_ylabel("normalized events", fontsize=fs)
        ax[i].tick_params(labelsize=fs)
        ax[i].legend(fontsize=fs, frameon=False)
        ax[i].set_yscale("log")
    fig.tight_layout()
    if save_fig:
        fig.savefig(plot_path + "residuals_all.png", format="png")
        plt.close(fig)
    else:
        fig.show()


def class_plots(
    pf_data,
    fs_data,
    tr_data,
    tr_pf_indices,
    pf_indices,
    tr_fs_indices,
    fs_indices,
    plot_path,
    save_fig=True,
    mask_fn=None,
):
    # make 2 confusion matrix tr-pf and tr-fs

    cut_off = 0.5

    tr_class_pf = (
        np.ones(np.sum([len(tr_pf_indices[i]) for i in range(len(tr_pf_indices))])) * -1
    )
    tr_class_fs = (
        np.ones(np.sum([len(tr_fs_indices[i]) for i in range(len(tr_fs_indices))])) * -1
    )
    pf_class_ = (
        np.ones(np.sum([len(pf_indices[i]) for i in range(len(pf_indices))])) * -1
    )
    fs_class_ = (
        np.ones(np.sum([len(fs_indices[i]) for i in range(len(fs_indices))])) * -1
    )

    curr_pf_len = 0
    curr_fs_len = 0
    pf_class = pf_data["class"]
    fs_class = fs_data["class"]
    tr_class = tr_data["class"]

    for i in tqdm(range(len(pf_class)), desc="class plots"):
        if len(pf_indices[i]) == 0 or len(fs_indices[i]) == 0:
            continue
        if mask_fn is not None:
            mask_tr = mask_fn(tr_data["idx"][i])
            mask_pf = mask_fn(pf_data["idx"][i])
            mask_fs = mask_fn(fs_data["idx"][i])
        else:
            mask_tr = np.ones_like(tr_data["pt"][i]) > 0
            mask_pf = np.ones_like(pf_data["pt"][i]) > 0
            mask_fs = np.ones_like(fs_data["pt"][i]) > 0

        tr_class_pf_i = tr_class[i][mask_tr][tr_pf_indices[i]]
        tr_class_fs_i = tr_class[i][mask_tr][tr_fs_indices[i]]
        pf_class_i = pf_class[i][mask_pf][pf_indices[i]]
        fs_class_i = fs_class[i][mask_fs][fs_indices[i]]

        tr_class_pf[curr_pf_len : curr_pf_len + len(tr_class_pf_i)] = tr_class_pf_i
        tr_class_fs[curr_fs_len : curr_fs_len + len(tr_class_fs_i)] = tr_class_fs_i
        pf_class_[curr_pf_len : curr_pf_len + len(pf_class_i)] = pf_class_i
        fs_class_[curr_fs_len : curr_fs_len + len(fs_class_i)] = fs_class_i

        curr_pf_len += len(tr_class_pf_i)
        curr_fs_len += len(tr_class_fs_i)

    class_labels = ["Ch had", "El", "Mu", "Neut had", "Phot"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=100, tight_layout=True)

    cm_pf = confusion_matrix(
        pf_class_, tr_class_pf, labels=[0, 1, 2, 3, 4], normalize="true"
    )
    df_cm = pd.DataFrame(cm_pf, index=class_labels, columns=class_labels)
    sn.heatmap(
        df_cm,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        cmap=sn.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
        cbar=False,
        ax=axs[0],
    )
    # axs[2].set_title('classes')
    axs[0].set_xlabel("Truth")
    axs[0].set_ylabel("Pflow")

    cm_fs = confusion_matrix(
        fs_class_, tr_class_fs, labels=[0, 1, 2, 3, 4], normalize="true"
    )
    df_cm = pd.DataFrame(cm_fs, index=class_labels, columns=class_labels)
    sn.heatmap(
        df_cm,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        cmap=sn.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
        cbar=False,
        ax=axs[1],
    )
    # axs[2].set_title('classes')
    axs[1].set_xlabel("Truth")
    axs[1].set_ylabel("Fastsim")

    if mask_fn is not None:
        mask_tr = mask_fn(np.concatenate(tr_data["idx"]))
        mask_pf = mask_fn(np.concatenate(pf_data["idx"]))
        mask_fs = mask_fn(np.concatenate(fs_data["idx"]))
    else:
        mask_tr = np.s_[:]
        mask_pf = np.s_[:]
        mask_fs = np.s_[:]
    tr_class_all = np.concatenate(tr_data["class"])[mask_tr]
    pf_class_all = np.concatenate(pf_data["class"])[mask_pf]
    fs_class_all = np.concatenate(fs_data["class"])[mask_fs]

    axs[2].hist(
        tr_class_all,
        bins=np.linspace(-0.5, 4.5, 6),
        histtype="step",
        color=c_tr,
        label="truth",
        density=True,
    )
    axs[2].hist(
        pf_class_all,
        bins=np.linspace(-0.5, 4.5, 6),
        histtype="step",
        color=c_pf,
        label="pflow",
        alpha=0.4,
        density=True,
    )
    axs[2].hist(
        fs_class_all,
        bins=np.linspace(-0.5, 4.5, 6),
        histtype="step",
        color=c_fs,
        label="fastsim",
        density=True,
    )
    axs[2].set_xlabel("Class", fontsize=fs)
    axs[2].set_ylabel("Events", fontsize=fs)
    axs[2].tick_params(labelsize=fs)
    axs[2].legend(fontsize=fs, frameon=False)
    axs[2].set_xticks([0, 1, 2, 3, 4], class_labels, fontsize=fs / 1.2)

    fig.tight_layout()
    if save_fig:
        fig.savefig(plot_path + "classes.png", format="png")
        plt.close(fig)
    else:
        fig.show()


def plot_hung_cost(
    pf_hung, fs_hung, plot_path, dict_for_comp={}, dr_match=False, save_fig=True
):
    if dr_match:
        bins = np.logspace(-6, 2, 200)
    else:
        bins = np.logspace(-3, 2, 100)

    fig = plt.figure(figsize=(6, 5), dpi=100, tight_layout=True)

    plt.hist(
        pf_hung, bins=bins, histtype="stepfilled", color=c_pf, label="pflow", alpha=0.4
    )
    plt.hist(fs_hung, bins=bins, histtype="step", color=c_fs, label="fastsim")
    plt.legend(frameon=False, fontsize=fs, loc="upper left")
    plt.tick_params(labelsize=fs)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Per event Hungarian cost", fontsize=fs)

    if save_fig:
        fig.savefig(plot_path + "hungarian_cost.png", format="png")
        plt.close(fig)
    else:
        fig.show()

    dict_for_comp["hung_cost"] = {"pf": pf_hung, "fs": fs_hung}