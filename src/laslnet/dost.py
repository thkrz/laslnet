import torch


def margin(window):
    half = window // 2
    return half, half - 1


def band(a):
    b = a.shape[-1]
    z = (a[..., ::2].sum(-1) - a[..., 1::2].sum(-1)) * (b**-0.5)
    return -z if b == 2 else z


def dost(im):
    N = im.shape[-1]
    m = N // 2
    n = N.bit_length() - 2
    IM = torch.fft.fft2(im) / (N * N)

    cols = [IM[..., 0]]
    for p in range(n):
        b = 1 << p
        cols.append(band(IM[..., b : 2 * b]))
    cols.append(IM[..., m])
    x = torch.stack(cols, dim=-1)

    rows = []
    for p in range(-n, n + 2):
        if p == 0:
            rows.append(x[..., 0, :])
        elif p == n + 1:
            rows.append(x[..., m, :])
        else:
            b = 1 << (abs(p) - 1)
            sl = slice(b, 2 * b) if p > 0 else slice(N - 2 * b + 1, N - b + 1)
            rows.append(band(x[..., sl, :].transpose(-2, -1)))

    z = torch.stack(rows, dim=-2)
    neg = torch.cat(
        (
            z[..., : 2 * n + 1, 1 : n + 1].flip((-2, -1)),
            z[..., 2 * n + 1 :, 1 : n + 1].flip(-1),
        ),
        dim=-2,
    ).conj()
    return torch.cat((neg, z), dim=-1)


@torch.no_grad()
def embed(data, field, window, device):
    voices = 2 * (window.bit_length() - 1)
    channels = voices * voices

    data = torch.as_tensor(data, dtype=torch.float32, device=device).contiguous()
    wins = data.unfold(0, window, 1).unfold(1, window, 1)
    out = torch.empty((field * field, channels), dtype=torch.complex64, device=device)
    batch = max(1, (256 * 1024 * 1024) // (4 * window * window))

    for i in range(0, len(out), batch):
        j = min(i + batch, len(out))
        indices = torch.arange(i, j, device=device)
        x = wins[indices // field, indices % field]
        x = x - x.mean((-1, -2), keepdim=True)
        z = dost(x)
        out[i:j] = z.reshape(j - i, channels)
        del x, z

    return out.reshape(field, field, channels).permute(2, 0, 1).contiguous()
