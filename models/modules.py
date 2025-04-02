"""Collection of pytorch modules that make up the common networks used in my
projects."""

import math
from typing import Optional, Union

import torch as T
import torch.nn as nn
from torch_utils import get_act, get_nrm


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str = "relu",
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
    ) -> None:
        """Init method for MLPBlock.

        Parameters
        ----------
        inpt_dim : int
            The number of features for the input layer
        outp_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of contextual features to concat to the inputs, by default 0
        n_layers : int, optional
            The number of transform layers in this block, by default 1
        act : str, optional
            A string indicating the name of the activation function, by default "lrlu"
        nrm : str, optional
            A string indicating the name of the normalisation, by default "none"
        drp : float, optional
            The dropout probability, 0 implies no dropout, by default 0
        do_res : bool, optional
            Add to previous output, only if dim does not change, by default 0
        do_bayesian : bool, optional
            If to fill the block with bayesian linear layers, by default False
        init_zeros : bool, optional,
            If the final layer weights and bias values are set to zero
            Does not apply to bayesian layers
        """
        super().__init__()

        # Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # If this layer includes an additive residual connection
        self.do_res = do_res and (inpt_dim == outp_dim)

        # Initialise the block layers as a module list
        self.block = nn.ModuleList()
        for n in range(n_layers):
            # Increase the input dimension of the first layer to include context
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim

            # Linear transform, activation, normalisation, dropout
            self.block.append(nn.Linear(lyr_in, outp_dim))

            # Initialise the final layer with zeros
            if init_zeros and n == n_layers - 1 and not do_bayesian:
                self.block[-1].weight.data.fill_(0)
                self.block[-1].bias.data.fill_(0)

            if act != "none":
                self.block.append(get_act(act))
            if nrm != "none":
                self.block.append(get_nrm(nrm, outp_dim))
            if drp > 0:
                self.block.append(nn.Dropout(drp))

    def forward(self, inpt: T.Tensor, ctxt: Optional[T.Tensor] = None) -> T.Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
            ctxt: The conditioning tensor, can be ignored
        """

        # Concatenate the context information to the input of the block
        if self.ctxt_dim and ctxt is None:
            raise ValueError(
                "Was expecting contextual information but none has been provided!"
            )
        temp = T.cat([inpt, ctxt], dim=-1) if self.ctxt_dim else inpt

        # Pass through each transform in the block
        for layer in self.block:
            temp = layer(temp)

        # Add the original inputs again for the residual connection
        if self.do_res:
            temp = temp + inpt

        return temp

    def __repr__(self) -> str:
        """Generate a one line string summing up the components of the
        block."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and
    context injection layers."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: Union[int, list] = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "relu",
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        do_bayesian: bool = False,
        output_init_zeros: bool = False,
    ) -> None:
        """Initialise the DenseNetwork.

        Parameters
        ----------
        inpt_dim : int
            The number of input neurons
        outp_dim : int, optional
            The number of output neurons. If none it will take from inpt or hddn,
            by default 0
        ctxt_dim : int, optional
            The number of context features. The context feature use is determined by
            ctxt_type, by default 0
        hddn_dim : Union[int, list], optional
            The width of each hidden block. If a list it overides depth, by default 32
        num_blocks : int, optional
            The number of hidden blocks, can be overwritten by hddn_dim, by default 1
        n_lyr_pbk : int, optional
            The number of transform layers per hidden block, by default 1
        act_h : str, optional
            The name of the activation function to apply in the hidden blocks,
            by default "lrlu"
        act_o : str, optional
            The name of the activation function to apply to the outputs,
            by default "none"
        do_out : bool, optional
            If the network has a dedicated output block, by default True
        nrm : str, optional
            Type of normalisation (layer or batch) in each hidden block, by default "none"
        drp : float, optional
            Dropout probability for hidden layers (0 means no dropout), by default 0
        do_res : bool, optional
            Use resisdual-connections between hidden blocks (only if same size),
            by default False
        ctxt_in_inpt : bool, optional
            Include the ctxt tensor in the input block, by default True
        ctxt_in_hddn : bool, optional
            Include the ctxt tensor in the hidden blocks, by default False
        do_bayesian : bool, optional
            Create the network with bayesian linear layers, by default False
        output_init_zeros : bool, optional
            Initialise the output layer weights as zeros

        Raises
        ------
        ValueError
            If the network was given a context input but both ctxt_in_inpt and
            ctxt_in_hddn were False
        """
        super().__init__()

        # Check that the context is used somewhere
        if ctxt_dim:
            if not ctxt_in_hddn and not ctxt_in_inpt:
                raise ValueError("Network has context inputs but nowhere to use them!")

        # We store the input, hddn (list), output, and ctxt dims to query them later
        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out

        # Necc for this module to work with the nflows package
        self.hidden_features = self.hddn_dim[-1]

        # Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
            act=act_h,
            nrm=nrm,
            drp=drp,
        )

        # All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm,
                        drp=drp,
                        do_res=do_res,
                    )
                )

        # Output block (optional and there is no normalisation, dropout or context)
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1],
                outp_dim=self.outp_dim,
                act=act_o,
                do_bayesian=do_bayesian,
                init_zeros=output_init_zeros,
            )

    def forward(self, inputs: T.Tensor, ctxt: Optional[T.Tensor] = None) -> T.Tensor:
        """Pass through all layers of the dense network."""

        # Reshape the context if it is available
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        # Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        # Pass through each hidden block
        for h_block in self.hidden_blocks:  # Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  # block was initialised with a ctxt dim

        # Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i + 1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += ">"
        string += str(self.input_block.outp_dim) + ">"
        if self.num_blocks > 1:
            string += ">".join(
                [
                    str(layer.out_features)
                    for hidden in self.hidden_blocks
                    for layer in hidden.block
                    if isinstance(layer, nn.Linear)
                ]
            )
            string += ">"
        if self.do_out:
            string += str(self.outp_dim)
        return string


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        max_period=10000,
        time_factor: float = 1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.time_factor = time_factor

        half = self.frequency_embedding_size // 2
        freqs = T.exp(
            -math.log(max_period) * T.arange(start=0, end=half, dtype=T.float32) / half
        )

        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t, time_factor: float = 1):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = t * time_factor

        args = t[:, None].float() * self.freqs[None]
        embedding = T.cat([T.cos(args), T.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = T.cat([embedding, T.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, time_factor=self.time_factor).type_as(t)
        t_emb = self.mlp(t_freq)
        return t_emb
