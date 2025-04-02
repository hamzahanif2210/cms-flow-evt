import torch
from tqdm import tqdm


@torch.no_grad()
def transfer(x_curr, d_curr, t_curr, t_next, dt=0):
    x_next = x_curr + d_curr * (t_next - t_curr - dt)
    return x_next


@torch.no_grad()
def runge_kutta(
    model,
    truth,
    fastsim,
    mask,
    global_data,
    t_list,
    dt=0,
):
    e_1 = model.forward(
        fastsim,
        truth,
        mask,
        timestep=t_list[0].expand(fastsim.shape[0]),
        global_data=global_data,
    )
    x_2 = transfer(fastsim, e_1, t_list[0], t_list[1], dt=dt)

    e_2 = model.forward(
        x_2,
        truth,
        mask,
        timestep=t_list[1].expand(fastsim.shape[0]),
        global_data=global_data,
    )
    x_3 = transfer(fastsim, e_2, t_list[0], t_list[1], dt=dt)

    e_3 = model.forward(
        x_3,
        truth,
        mask,
        timestep=t_list[1].expand(fastsim.shape[0]),
        global_data=global_data,
    )
    x_4 = transfer(fastsim, e_3, t_list[0], t_list[2], dt=dt)

    e_4 = model.forward(
        x_4,
        truth,
        mask,
        timestep=t_list[2].expand(fastsim.shape[0]),
        global_data=global_data,
    )
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et, e_1


@torch.no_grad()
def gen_order_4(
    model,
    truth,
    fastsim,
    mask,
    global_data,
    t,
    t_next,
    ets,
    dt=0,
):
    t_list = [t, (t + t_next) / 2, t_next]
    if len(ets) > 2:
        deriv_ = model.forward(
            fastsim,
            truth,
            mask,
            timestep=t.expand(fastsim.shape[0]),
            global_data=global_data,
        )
        ets.append(deriv_)
        deriv = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        deriv, e_1 = runge_kutta(
            model,
            truth,
            fastsim,
            mask,
            global_data,
            t_list,
            dt=dt,
        )
        ets.append(e_1)
    if len(ets) > 4:
        ets.pop(0)
    fastsim_next = transfer(fastsim, deriv, t, t_next, dt=dt)
    return fastsim_next


@torch.no_grad()
def pndm_sampler(
    model,
    truth,
    pflow_shape,
    mask,
    global_data,
    n_steps,
    dt=0,
    save_seq=False,
    zero_init_padded=True,
    reverse_time=False,
):
    seq = []
    device = truth.device
    fastsim = torch.randn(pflow_shape, device=device)
    if zero_init_padded:
        fastsim[~mask[..., 1]] = 0

    t_steps = torch.linspace(0, 1, n_steps, device=device)
    t_steps = torch.cat([t_steps, torch.ones_like(t_steps)[:1]]).to(device)
    if reverse_time:
        t_steps = torch.flip(t_steps, [0])
    ets = []
    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=n_steps
    ):
        fastsim = gen_order_4(
            model,
            truth,
            fastsim,
            mask,
            global_data,
            t_cur,
            t_next,
            ets,
            dt=dt,
        )
        if save_seq:
            seq.append(fastsim.cpu())
    if save_seq:
        seq = torch.stack(seq)
    return fastsim.cpu(), seq