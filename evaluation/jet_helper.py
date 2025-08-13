import vector as vec
import fastjet as fj
import numpy as np
import tqdm as tqdm
import gc
import energyflow as ef


class Jet(object):
    def __init__(self, fj_jet, R, calc_substructure=False):
        self.fj_jet = fj_jet
        self.R = R
        self.nconstituents = len(self.constituents())
        self.constituents_pt = np.array([c.pt() for c in self.constituents()])
        self.constituents_eta = np.array([c.eta() for c in self.constituents()])
        self.constituents_phi = np.array(
            [
                c.phi() if c.phi() <= np.pi else c.phi() - 2 * np.pi
                for c in self.constituents()
            ]
        )
        self.constituents_m = np.array([c.m() for c in self.constituents()])
        self.constituents_idx = np.array([c.user_index() for c in self.constituents()])
        self.constituents_class = None
        self.pt_order_constituents()

        self.dR_matrix = None
        self.ecf = {0: 1, 1: -1, 2: -1, 3: -1}
        self.substructure = {"c2": np.nan, "d2": np.nan}

        if calc_substructure:
            self.calc_substructure()

    def __getattr__(self, name):
        if name in [
            "pt",
            "eta",
            "phi",
            "phi_std",
            "e",
            "m",
            "constituents",
            "px",
            "py",
            "pz",
            "E",
        ]:
            return getattr(self.fj_jet, name)
        else:
            return getattr(self, name)

    def pt_order_constituents(self):
        idx = np.argsort(self.constituents_pt)
        self.constituents_pt = self.constituents_pt[idx]
        self.constituents_eta = self.constituents_eta[idx]
        self.constituents_phi = self.constituents_phi[idx]
        self.constituents_m = self.constituents_m[idx]
        self.constituents_idx = self.constituents_idx[idx]
        if self.constituents_class is not None:
            self.constituents_class = self.constituents_class[idx]

    def set_constituents_class(self, class_arr):
        self.constituents_class = [class_arr[i] for i in self.constituents_idx]

    def get_dR_matrix(self):
        eta_matrix = np.repeat(
            self.constituents_eta.reshape(1, self.nconstituents),
            self.nconstituents,
            axis=0,
        )
        phi_matrix = np.repeat(
            self.constituents_phi.reshape(1, self.nconstituents),
            self.nconstituents,
            axis=0,
        )
        eta_matrix = eta_matrix - eta_matrix.T
        phi_matrix = np.abs(phi_matrix - phi_matrix.T)
        dR_matrix = (eta_matrix**2 + phi_matrix**2) ** 0.5

        self.dR_matrix = dR_matrix

    def calc_substructure(self):
        d2_calc = ef.D2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
        c2_calc = ef.C2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)

        pt_eta_phi_m = np.stack(
            [
                self.constituents_pt,
                self.constituents_eta,
                self.constituents_phi,
                self.constituents_m,
            ],
            axis=1,
        )

        self.substructure["d2"] = d2_calc.compute(pt_eta_phi_m)
        self.substructure["c2"] = c2_calc.compute(pt_eta_phi_m)


def get_jet_definition(algo, radius):
    if algo == "genkt":
        jet_definition = fj.JetDefinition(fj.ee_genkt_algorithm, radius, -1.0)
    elif algo == "antikt":
        jet_definition = fj.JetDefinition(fj.antikt_algorithm, radius)
    else:
        raise NotImplementedError(f"Jet algorithm {algo} not implemented!")

    return jet_definition


def get_cluster_sequence(jet_definition, four_vectors, user_indices=None):
    pj_array = []

    for i, part in enumerate(four_vectors):
        pj = fj.PseudoJet(part.px.item(), part.py.item(), part.pz.item(), part.E.item())
        if user_indices is not None:
            pj.set_user_index(user_indices[i])
        else:
            pj.set_user_index(i)
        pj_array.append(pj)

    cs = fj.ClusterSequence(pj_array, jet_definition)

    return cs


def get_pt_sorted_jets(
    four_vectors,
    algo,
    radius,
    pt_min,
    n_const_min,
    eta_max,
    user_indices=None,
    constituents_classes=None,
):
    jet_definition = get_jet_definition(algo, radius)

    cluster_sequence = get_cluster_sequence(
        jet_definition, four_vectors, user_indices=user_indices
    )

    jets = cluster_sequence.inclusive_jets(ptmin=pt_min)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, radius) for j in jets]

    if constituents_classes is not None:
        for jet in jets:
            jet.set_constituents_class(constituents_classes)

    jets = [j for j in jets if j.nconstituents >= n_const_min]
    jets = [j for j in jets if abs(j.eta()) < eta_max]

    return jets