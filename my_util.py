import os
import numpy as np
import logging


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def filter_events(event_data, start, end):
    ## filter events based on temporal dimension
    x = event_data['x'][event_data['t'] >= start]
    y = event_data['y'][event_data['t'] >= start]
    p = event_data['p'][event_data['t'] >= start]
    t = event_data['t'][event_data['t'] >= start]

    x = x[t <= end]
    y = y[t <= end]
    p = p[t <= end]
    t = t[t <= end]
    return x, y, p, t


def filter_events_by_space(key, x1, x2, x3, start, end):
    ## filter events based on spatial dimension
    # start inclusive and end exclusive
    mask = (key >= start) & (key < end)
    new_x1 = x1[mask]
    new_x2 = x2[mask]
    new_x3 = x3[mask]
    new_key = key[mask]

    return new_key, new_x1, new_x2, new_x3


def filter_events_for_rs(t, y, x, p, key_t, ts):
    ## filter events based on spatial dimension
    # start inclusive and end exclusive
    mask = (t >= np.minimum(key_t[y], ts[y])) & (t < np.maximum(key_t[y], ts[y]))
    new_x1 = y[mask]
    new_x2 = x[mask]
    new_x3 = p[mask]
    new_key = t[mask]

    return new_key, new_x1, new_x2, new_x3



def e2f_detail(event, eframe, ts, key_t, noise, interval=0):
    T, C, H, W = eframe.shape
    eframe = eframe.ravel()
    # if key_t < ts:
    ## reverse event time & porlarity
    x, y, p, t = event['x'], event['y'], event['p'], event['t']
    t, y, x, p = filter_events_for_rs(t, y, x, p, key_t, ts)  # filter events by time
    p = p * 2 - 1
    new_t = np.abs(t - ts[y])

    # if not interval:
    mask = t - ts[y] < 0
    p[mask] = -p[mask]
    idx = np.floor(new_t * T / np.abs(key_t[y] - ts[y])).astype(int)
    idx[idx == T] -= 1
    # assert(idx.max()<T)
    p[p == -1] = 0  # reversed porlarity
    np.add.at(eframe, x + y * W + p * W * H + idx * W * H * C, 1)

    assert noise >= 0 and noise <= 1
    if noise > 0:
        num_noise = int(noise * len(t))
        img_size = (H, W)
        noise_x = np.random.randint(0, img_size[1], (num_noise, 1))
        noise_y = np.random.randint(0, img_size[0], (num_noise, 1))
        noise_p = np.random.randint(0, 2, (num_noise, 1))
        noise_t = np.random.randint(0, idx + 1, (num_noise, 1))
        # add noise
        np.add.at(eframe, noise_x + noise_y * W + noise_p * W * H + noise_t * W * H * C, 1)

    eframe = np.reshape(eframe, (T, C, H, W))

    return eframe


def event2frame(event, imgsize, low_limit, high_limit1, high_limit2, num_frame, noise, interval=0):
    ## convert event streams to [T, C, H, W] event tensor, C=2 indicates polarity
    preE = np.zeros((num_frame, 2, imgsize[0], imgsize[1]))
    postE = np.zeros((num_frame, 2, imgsize[0], imgsize[1]))
    total_time = np.concatenate((low_limit,high_limit1,high_limit2),0)
    if event['t'].shape[0] > 0:
        preE = e2f_detail(event, preE, low_limit, high_limit1, noise)
        postE = e2f_detail(event, postE, low_limit, high_limit2, noise)

    preE = preE.reshape(2 * num_frame, imgsize[0], imgsize[1])
    postE = postE.reshape(2 * num_frame, imgsize[0], imgsize[1])
    return preE, postE

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
