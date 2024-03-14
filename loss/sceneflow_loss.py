import torch

def sceneflow_loss_func(flow_preds, flow_gt, 
                   gamma=0.9,
                   **kwargs,
                   ):

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):

        diff = (flow_preds[i] - flow_gt).abs() # [B, 2, N]
        epe_l1 = torch.pow(diff.sum(dim=1) + 0.01, 0.4).mean(dim=-1)
    
        i_weight = gamma ** (n_predictions - i - 1)
        flow_loss += i_weight * epe_l1

    # compute endpoint error
    diff = (flow_preds[-1] - flow_gt).abs()
    epe3d_map = torch.sqrt(torch.sum(diff ** 2, dim=1)) # [B, N]
    epe3d_bat = epe3d_map.mean(dim=1)

    # compute 5cm accuracy
    acc5_3d_map = (epe3d_map < 0.05).float() * 1.
    acc5_3d_bat = acc5_3d_map.mean(dim=1)

    # compute 5cm accuracy
    acc10_3d_map = (epe3d_map < 0.10).float() * 1.
    acc10_3d_bat = acc10_3d_map.mean(dim=1)

    # compute outlier
    outlier_3d_map = (epe3d_map > 0.30).float() * 1.
    outlier_3d_bat = outlier_3d_map.mean(dim=1)

    metrics = {
        'epe3d': epe3d_bat.mean().item(),
        'acc3d_5cm':acc5_3d_bat.mean().item(),
        'acc3d_10cm':acc10_3d_bat.mean().item(),
        'outlier_30cm':outlier_3d_bat.mean().item(),
    }

    return flow_loss, metrics
