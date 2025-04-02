from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

import models.norms as norms

Tensors: TypeAlias = dict[str, Tensor]

def projection_packed(
    q: Tensor,
    kv: Tensor | None,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple:
    """Efficient input projection for MHA when using a single linear layer.

    Essentially the same as torch.nn.functional._in_projection_packed
    But here we use chunk which is 40x faster than unflatten
    Not sure why they don't use chunk in the original implementation...

    Parameters
    ----------
    q : Tensor
        The queries tensor of shape (batch, q_len, dim).
    kv : Tensor | None
        The keys and values tensor of shape (batch, kv_len, dim).
    weight : Tensor
        The packed weight tensor of the input lienar projection with shape (3 * dim, dim).
    bias : Tensor | None
        The optional packed bias tensor of the input linear projection with shape (3 * dim).

    Returns
    -------
    q_proj, k_proj, v_proj : tuple
        The projected queries, keys, and values tensors.
    """
    # If the q tensor is the only input, then we assume we are doing self-attention.
    # This is made (slightly) faster by using a single linear layer, then chunking rather than
    # three seperate linear layers processed one at a time.
    if kv is None:
        return F.linear(q, weight, bias).chunk(3, dim=-1)

    # If the kv tensor is present, then we are doing cross-attention.
    # This means we must project the q and kv tensors seperately.
    # The kv linear layer can remain packed, allowing us to project together then chunk,
    # using the same trick as above. We must however first seperate weights (and biases if present)
    # of the linear layers for the q and kv parts. We use torch.split which returns a veiw of the
    # original tensor so this step doesnt required any extra memory or much time.
    dim = q.size(-1)
    w_q, w_kv = weight.split([dim, dim * 2])
    b_q, b_kv = bias.split([dim, dim * 2]) if bias is not None else (None, None)

    # Now we can do the seperate projections
    q_proj = F.linear(q, w_q, b_q)
    k_proj, v_proj = F.linear(kv, w_kv, b_kv).chunk(2, dim=-1)
    return q_proj, k_proj, v_proj

def merge_masks(
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Tensor,
) -> BoolTensor | None:
    """Create a full attention mask which incorporates the padding information.

    Using pytorch transformer convention for padding
        False: Real node
        True:  Zero padded

    Using pytorch transformer convention for attention mask
        False:  Not allowed in attention mechanism
        True:   Allowed in attention mechanism

    Designing attention mask such that padded tokens can't send information.
    But they can receive them.
    This prevents Nans in the attention scores caused by the softmax

    Parameters
    ----------
    kv_mask : BoolTensor | None
        Mask for the keys and values, of shape (batch, kv_len).
    attn_mask : BoolTensor | None
        Full attention mask, of shape (batch, q_len, kv_len).
    q_shape : Size
        Shape of the queries tensor, (batch, q_len, dim).
    """
    # Create the full mask which combines the attention and padding masks
    mask = None

    # if the kv_mask mask exists, ensure that padded tokens never send information
    if kv_mask is not None:
        mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        mask = ~mask  # convert the mask such that True is a valid token

    # include the attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            mask = attn_mask if mask is None else attn_mask & mask
        else:
            mask = attn_mask if mask is None else attn_mask + (~mask) * (-1e9)

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if mask is not None:
        mask = mask.unsqueeze(1)

    return mask


def repeat_kv(keys: Tensor, values: Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def change_attn_backends(module: nn.Module, backend: str) -> None:
    """Recursively change the attention backend of a module and all its children.

    Used primarily for switching back to torch-math for ONNX exports.
    """
    for child in module.children():
        change_attn_backends(child, backend)
        if isinstance(child, Attention):
            child.set_backend(backend)

def torch_attn(
    q: Tensor, k: Tensor, v: Tensor, mask: BoolTensor, dropout: float, backend: str
) -> Tensor:
    """Torch dot product attention with a switchable backend."""
    backends = [SDPBackend.MATH]
    if backend == "torch-meff":
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    elif backend == "torch-flash":
        backends.append(SDPBackend.FLASH_ATTENTION)
    elif backends == "torch-cudnn":
        backends.append(SDPBackend.CUDNN_ATTENTION)
    with sdpa_kernel(backends):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout
        )


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        attn_type: str = "torch-meff",
        dropout: float = 0.0,
        bias: bool = True,
        do_qk_norm: bool = False,
        do_selfattn: bool = False,
    ) -> None:
        """Multihead attention module.

        Parameters
        ----------
        embed_dim : int
            Dimension of the input.
        num_heads : int
            Number of attention heads.
        attn_type : str, optional
            Name of backend kernel to use.
        dropout : float, optional
            Dropout rate.
        bias : bool, optional
            Whether to include bias terms.
        do_qk_norm : bool, optional
            Whether to normalise the queries and keys.
        do_selfattn : bool, optional
            Whether to perform self-attention.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Dim not div by the number of heads!"
        assert attn_type in {
            "torch-flash",
            "torch-math",
            "torch-meff",
            "torch-cudnn",
            "torch-flex",
        }, "Invalid attention type!"

        self.attn_type = attn_type
        if attn_type == "torch-flex":
            self.attn_fn = torch.compile(flex_attention)
        else:
            self.attn_fn = torch_attn
        self.do_selfattn = do_selfattn
        self.do_qk_norm = do_qk_norm

        # Attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias

        if do_qk_norm:
            self.q_norm = norms.RMSNorm(self.head_dim)
            self.k_norm = norms.RMSNorm(self.head_dim)

        # Better parallelism for self-attention when using parameters directly
        if do_selfattn:
            self.in_proj_weight = nn.Parameter(
                torch.randn(3 * embed_dim, embed_dim) / embed_dim**0.5
            )
            if self.bias:
                self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
            else:
                self.in_proj_bias = None
        else:
            self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the parameters."""
        if self.do_selfattn:
            nn.init.xavier_uniform_(self.in_proj_weight)
            if self.bias:
                nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def _torch_forward(
        self,
        x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
        attn_bias: Tensor | None = None,
    ):
        "Attention using pytorch"
        # Otherwise perform standard attention
        _, S, D = x.size()

        # input projections -> B, S, D
        if self.do_selfattn:
            q, k, v = projection_packed(x, None, self.in_proj_weight, self.in_proj_bias)
        else:
            q, k, v = self.q_linear(x), self.k_linear(kv), self.v_linear(kv)

        # transform tensors to (B, Nh, S, Hd)
        shape = (-1, S, self.num_heads, self.head_dim)  # Dont use S for cross attn
        # shape = (B, -1, self.num_heads, self.head_dim)  # Dont use S for cross attn
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        # Optionally normalise the queries and keys
        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.do_v_norm:
            v = self.v_norm(v)

        # run attention
        s_mask = mask if kv is None else kv_mask  # Who is sending, x or kv
        mask = merge_masks(s_mask, attn_mask, q.shape)
        dropout = self.dropout if self.training else 0.0
        a_out = torch_attn(q, k, v, mask, dropout, self.attn_type)

        if attn_bias is not None:
            a_out = a_out + attn_bias.unsqueeze(1) @ v

        # recombine heads
        a_out = a_out.transpose(1, 2).contiguous().view(_, S, D)

        # mix with final linear layer
        return self.out_proj(a_out)

    def _flex_forward(
        self,
        x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
    ):
        "Attention using pytorch"
        # Otherwise perform standard attention
        B, S, D = x.shape

        # input projections -> B, S, D
        if self.do_selfattn:
            q, k, v = projection_packed(x, None, self.in_proj_weight, self.in_proj_bias)
        else:
            q, k, v = self.q_linear(x), self.k_linear(kv), self.v_linear(kv)

        # transform tensors to (B, Nh, S, Hd)
        shape = (B, -1, self.num_heads, self.head_dim)  # Dont use S for cross attn
        q, k, v = (t.view(shape).transpose(1, 2).contiguous() for t in (q, k, v))

        # Optionally normalise the queries and keys
        if self.do_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        a_out = self.attn_fn(q, k, v, block_mask=mask)

        # recombine heads
        a_out = a_out.transpose(1, 2).contiguous().view(B, S, D)

        # mix with final linear layer
        return self.out_proj(a_out)

    def forward(
        self,
        x: Tensor,
        kv: Tensor | None = None,
        mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BoolTensor | None = None,
        attn_bias: Tensor | None = None,
    ) -> Tensor:
        """Attention forward pass.

        Parameters
        ----------
        x : Tensor
            The pointcloud of shape (batch, x_len, dim).
        kv : Tensor
            Optional second pointcloud for cross-attn with shape (batch, kv_len, dim).
        mask : BoolTensor, optional
            Mask for the pointcloud x, by default None.
        kv_mask : BoolTensor, optional
            Mask the kv pointcloud, by default None.
        attn_mask : BoolTensor, optional
            Full attention mask, by default None.
        attn_bias : Tensor, optional
            Optional bias to add to the attention scores, by default None.
        pos_trans_q : Tensor, optional
            Positional transformation for the queries, by default None.
        pos_trans_k : Tensor, optional
            Positional transformation for the keys, by default None.

        Returns
        -------
        Tensor
            Output of shape (batch, x_len, dim).
        """
        if self.attn_type == "torch-flex":
            return self._flex_forward(
                x=x,
                kv=kv,
                mask=kv_mask,
            )
        return self._torch_forward(
            x=x,
            kv=kv,
            mask=mask,
            kv_mask=kv_mask,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
        )
