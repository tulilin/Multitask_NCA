import numpy as np
import torch

def hist_match(image_a, image_b):
    h1, w1 = image_a.size()
    h2, w2 = image_b.size()

    I = image_a.reshape(-1)
    J = image_b.reshape(-1)

    num_pixel_a = torch.bincount(I.int())
    num_pixel_b = torch.bincount(J.int())

    prob_pixel_a = num_pixel_a / (h1 * w1 * 1.0)
    prob_pixel_b = num_pixel_b / (h2 * w2 * 1.0)

    cumu_pixel_a = torch.cumsum(prob_pixel_a, dim=0)
    cumu_pixel_b = torch.cumsum(prob_pixel_b, dim=0)

    sc_min = torch.abs(cumu_pixel_a.reshape(-1, 1) - cumu_pixel_b)

    hist_m = torch.argmin(sc_min, dim=1)

    I_new = torch.gather(hist_m, 0, I.long())
    I_new[I == 0] = 0

    image_a_match = I_new.reshape(h1, w1)

    return image_a_match