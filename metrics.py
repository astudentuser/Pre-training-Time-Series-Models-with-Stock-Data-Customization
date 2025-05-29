import numpy as np


def return_eval(all_pred_close, all_labels, all_mask, topks):
    # all_pred_close [trading_days, stock_num] ([252, 1026])
    # all_labels [trading_days, stock_num, class_num] ([252, 1026, 2])
    cul_ratios = []
    srs = []
    for topk in topks:
        cul_ratio = 1.0
        sr= []
        for i in range(len(all_pred_close)):
            pred_close = all_pred_close[i].cpu().detach().numpy().squeeze()
            labels = all_labels[i].cpu().detach().numpy()
            mask = all_mask[i].cpu().detach().numpy()
            base_price = labels[:, 0]
            obj_price = labels[:, 1]
            pred_ratio = (pred_close - base_price) / base_price
            obj_ratio = (obj_price - base_price) / base_price

            rank_pre = np.argsort(pred_ratio)[::-1]
            pre_topk = []
            for j in range(len(pred_close)):
                if mask[rank_pre[j]] < 0.5:
                    continue
                if len(pre_topk) < topk:
                    pre_topk.append(rank_pre[j])
                if len(pre_topk) == topk:
                    break

            topk_return_ratio = 0
            for pre in pre_topk:
                topk_return_ratio += obj_ratio[pre]
            topk_return_ratio = topk_return_ratio / topk

            cul_ratio *= (1 + topk_return_ratio)
            sr.append(topk_return_ratio)
        cul_ratio = cul_ratio - 1
        sr = np.array(sr)
        sr = (np.mean(sr) / np.std(sr)) * 15.87  # To annualize
        cul_ratios.append(cul_ratio)
        srs.append(sr)
    return cul_ratios, srs

