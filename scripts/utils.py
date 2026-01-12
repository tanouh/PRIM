import numpy as np

import torch

@torch.no_grad()
def extract_embeddings(model, loader, device):
    embeddings = []
    labels = []
    paths = []

    for imgs, lbls, pths in loader:
        imgs = imgs.to(device)
        z = model.forward_once(imgs)
        embeddings.append(z.cpu())
        labels.extend(lbls)
        paths.extend(pths)

    return torch.cat(embeddings), np.array(labels), paths