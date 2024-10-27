from datasets import *
from utils import *

set_seed(2603)
n_samples=20

def retrieve_scope_samples(k=3, s=3):
    model = 'esm2-150M'
    scope_data_avg = torch.load(f'{DATA_DIR["scope"]}/train_{model}_avg.pt')
    scope_data_cls = torch.load(f'{DATA_DIR["scope"]}/train_{model}_cls.pt')
    scope_data_full = torch.load(f'{DATA_DIR["scope"]}/train_{model}.pt')
    samples = []
    n_group = 1
    folds = np.random.choice(list(scope_data_avg['meta_info']['folds'].keys()), n_samples // n_group, replace=False)
    for i in range(n_samples // n_group):
        sf = np.random.choice(list(scope_data_avg['meta_info']['folds'][folds[i]]))
        for j in range(n_group):
            f = np.random.choice(list(scope_data_avg['meta_info']['superfams'][sf]))
            sid = np.random.choice(list(scope_data_avg['meta_info']['fams'][f]))
            samples.append(sid)
    avg_pooled_samples = []
    cls_pooled_samples = []
    bom_pooled_samples = []
    full_embedding_samples = []

    for i, sid in enumerate(samples):
        _, _, _, id = scope_data_avg['meta_info']['membership'][sid]
        avg_pooled_samples.append(scope_data_avg['seq_emb'][id])
        cls_pooled_samples.append(scope_data_cls['seq_emb'][id])
        full_embedding_samples.append(scope_data_full['seq_emb'][id])
        bom_pooled_samples.append(full_embedding_samples[-1].unfold(0, k, s).mean(dim=-1))
        
    avg_pooled_samples = torch.stack(avg_pooled_samples)
    cls_pooled_samples = torch.stack(cls_pooled_samples)
    return avg_pooled_samples, cls_pooled_samples, bom_pooled_samples, full_embedding_samples

def pairwise_cosim_avg_cls(avg_pooled_samples, cls_pooled_samples):
    avg_cosim = pairwise_cosine_similarity(avg_pooled_samples) + torch.eye(n_samples)
    cls_cosim = pairwise_cosine_similarity(cls_pooled_samples) + torch.eye(n_samples)
    return avg_cosim, cls_cosim

def pairwise_cosim_bom(bom_pooled_samples):
    bom_cosim = torch.zeros(len(bom_pooled_samples), len(bom_pooled_samples))
    for i, s1 in enumerate(bom_pooled_samples):
        for j, s2 in enumerate(bom_pooled_samples):
            maxlen = min(s1.shape[0], s2.shape[0])
            bom_cosim[i, j] = F.cosine_similarity(s1[:maxlen], s2[:maxlen]).min()
    return bom_cosim

def token_to_pooled_cosim(avg_pooled_samples, cls_pooled_samples, bom_pooled_samples, full_embedding_samples):
    token_to_avg_cosim = []
    token_to_cls_cosim = []
    token_to_bom_cosim = []
    for i in range(n_samples):
        token_to_avg_cosim.append(pairwise_cosine_similarity(full_embedding_samples[i][1:-1], avg_pooled_samples[i:i+1]).flatten().numpy())
        token_to_cls_cosim.append(pairwise_cosine_similarity(full_embedding_samples[i][1:-1], cls_pooled_samples[i:i+1]).flatten().numpy())
        token_to_bom_cosim.append(pairwise_cosine_similarity(full_embedding_samples[i][1:-1], bom_pooled_samples[i]).max(dim=-1)[0].numpy())
    return token_to_avg_cosim, token_to_cls_cosim, token_to_bom_cosim

def infoloss_avg_cls(pooled_samples, full_embedding_samples):
    return torch.stack([
        torch.cdist(pooled_samples[i:i+1], full_embedding_samples[i]).mean()
        for i in range(n_samples)
    ])

def infoloss_bom(bom_pooled_samples, full_embedding_samples):
    bom_infoloss = []
    for i in range(n_samples):
        infoloss = torch.cdist(bom_pooled_samples[i], full_embedding_samples[i])
        bom_infoloss.append(torch.min(infoloss, dim=0).values.mean())
    return torch.stack(bom_infoloss)

def plot_infoloss_vs_variance(cls_infoloss, avg_infoloss, bom_infoloss, full_embedding_samples):
    var = [torch.std(s, dim=0).mean().item() for s in full_embedding_samples]
    fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))
    axes.scatter(var, cls_infoloss, label='CLS-Pooling')
    axes.scatter(var, avg_infoloss, label='Avg-Pooling')
    axes.scatter(var, bom_infoloss, label='BoM-Pooling')
    axes.set_ylabel('Pooling Info Loss', fontsize=10)
    axes.set_xlabel('Mean Token Standard Deviation', fontsize=10)
    axes.legend(loc='upper left', fontsize=8)
    # axes.set_xlim([1.95, 12.05])
    # axes.set_xticks([2, 4, 6, 8, 10, 12])
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2)
    fig.savefig(f'./artifact/visualization/infoloss_variance.png')

def visualize_v1(avg_cosim, cls_cosim, token_to_avg_cosim, token_to_cls_cosim, token_to_bom_cosim):
    vmin = min(np.min(avg_cosim), np.min(cls_cosim))
    vmax = max(np.max(avg_cosim), np.max(cls_cosim))

    for t in range(n_samples):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 2.5), gridspec_kw={'width_ratios': [1, 1]})
        xticks, yticks = np.array([4, 9, 14, 19]), np.array([4, 9, 14, 19])
        xlabel, ylabel = xticks + 1, yticks + 1
        im0 = axes[0].imshow(avg_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Avg-Pooling Cosine Similarity', fontsize=10)
        axes[0].set_xticks(xticks, labels=xlabel)
        axes[0].set_yticks(yticks, labels=ylabel)

        im1 = axes[1].imshow(cls_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('CLS-Pooling Cosine Similarity', fontsize=10)
        axes[1].set_xticks(xticks, labels=xlabel)
        axes[1].set_yticks(yticks, labels=ylabel)

        # add space for colour bar and cosim plot
        fig.subplots_adjust(left=0.1, right=0.44)
        cbar_ax = fig.add_axes([0.02, 0.12, 0.011, 0.74])
        fig.colorbar(im1, cax=cbar_ax)
        # cbar_ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

        # cosim plot
        ax2 = fig.add_axes([0.5, 0.11, 0.48, 0.77]) 
        ax2.set_title('Cosine Similarity to Pooled Representation', fontsize=10)
        ax2.plot(np.arange(token_to_avg_cosim[t].shape[0]), token_to_avg_cosim[t], label='Avg')
        ax2.plot(np.arange(token_to_cls_cosim[t].shape[0]), token_to_cls_cosim[t], label='CLS')
        ax2.plot(np.arange(token_to_bom_cosim[t].shape[0]), token_to_bom_cosim[t], label='BoM')
        ax2.set_ylabel('Cosine Similarity', fontsize=10)
        ax2.set_xlabel('AA Position', fontsize=10)
        ax2.legend(ncol=3, fontsize=10)
        fig.savefig(f'./artifact/visualization/embedding_visualization_{t}.png')

def cosim_plot(avg_cosim, cls_cosim):
    vmin = min(np.min(avg_cosim), np.min(cls_cosim))
    vmax = max(np.max(avg_cosim), np.max(cls_cosim))
    xticks, yticks = np.array([4, 9, 14, 19]), np.array([4, 9, 14, 19])
    xlabel, ylabel = xticks + 1, yticks + 1
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

    axes[0].imshow(avg_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_xlabel('Avg-Pooling Cosine Similarity', fontsize=10)
    axes[0].set_xticks(xticks, labels=xlabel)
    axes[0].set_yticks(yticks, labels=ylabel)

    im1 = axes[1].imshow(cls_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_xlabel('CLS-Pooling Cosine Similarity', fontsize=10)
    axes[1].set_xticks(xticks, labels=xlabel)
    axes[1].set_yticks(yticks, labels=ylabel)

    fig.subplots_adjust(left=0.2, right=0.95, wspace=0.3, bottom=0.2)
    cbar_ax = fig.add_axes([0.02, 0.21, 0.02, 0.65])
    fig.colorbar(im1, cax=cbar_ax)
    fig.savefig(f'./artifact/visualization/pairwise_cosim.png')

def infoloss_plot(avg_infoloss, cls_infoloss, bom_infoloss):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    axes[0].scatter(avg_infoloss, bom_infoloss)
    axes[0].plot([3, 10], [3, 10], color='black', linestyle='--')
    axes[0].set_xlabel('Avg-Pooling Info Loss', fontsize=10)
    axes[0].set_ylabel('BoM-Pooling Info Loss', fontsize=10)

    axes[1].scatter(cls_infoloss, bom_infoloss)
    axes[1].plot([3, 10], [3, 10], color='black', linestyle='--')
    axes[1].set_xlabel('CLS-Pooling Info Loss', fontsize=10)
    # axes[1].set_ylabel('BoM-Pooling Info Loss', fontsize=10)

    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.3, bottom=0.2)
    fig.savefig(f'./artifact/visualization/infoloss_scatter.png')

def retrieve_fluo_lohi(k=3, s=3):
    data = torch.load(f'{DATA_ROOT}/fluorescence/fluorescence_esm2-150M.pt')
    n_samples = 10
    high_act = torch.topk(data['y_test'], k=n_samples * 2).indices.numpy()
    low_act = torch.topk(-data['y_test'], k=n_samples * 2).indices.numpy()

    x_high = [data['x_test'][idx] for idx in high_act if data['x_test'][idx].shape[0] == 239]
    x_low = [data['x_test'][idx] for idx in low_act if data['x_test'][idx].shape[0] == 239]
    x_high = x_high[:n_samples]
    x_low = x_low[:n_samples]

    x_ave_high = torch.stack([x.mean(dim=0) for x in x_high])
    x_ave_low = torch.stack([x.mean(dim=0) for x in x_low])
    x_ave = torch.cat([x_ave_high, x_ave_low])
    dist_ave = pairwise_cosine_similarity(x_ave, x_ave)
    pp_ave = dist_ave[:n_samples, :n_samples].flatten().numpy()
    pn_ave = dist_ave[:n_samples, n_samples:].flatten().numpy()
    dist_ave = dist_ave.numpy()

    x_cls_high = torch.stack([x[0] for x in x_high])
    x_cls_low = torch.stack([x[0] for x in x_low])
    x_cls = torch.cat([x_cls_high, x_cls_low])
    dist_cls = pairwise_cosine_similarity(x_cls, x_cls)
    pp_cls = dist_cls[:n_samples, :n_samples].flatten().numpy()
    pn_cls = dist_cls[:n_samples, n_samples:].flatten().numpy()
    dist_cls = dist_cls.numpy()

    x_bom_high = [x.unfold(0, k, s).mean(dim=-1) for x in x_high]
    x_bom_low = [x.unfold(0, k, s).mean(dim=-1) for x in x_low]
    x_bom = x_bom_high + x_bom_low
    dist_bom = torch.zeros(2 * n_samples, 2 * n_samples)
    for i, x1 in enumerate(x_bom):
        for j, x2 in enumerate(x_bom):
            dij = pairwise_cosine_similarity(x1, x2)
            for t in range(min(x1.shape[0], x2.shape[0])):
                dij[t][t] -= 1.
            dist_bom[i,j] = dij.max(dim=-1).values.min()

    pp_bom = dist_bom[:n_samples, :n_samples].flatten().numpy()
    pn_bom = dist_bom[:n_samples, n_samples:].flatten().numpy()
    dist_bom = dist_bom.numpy()

    torch.save({
        'cls': [pp_cls, pn_cls],
        'ave': [pp_ave, pn_ave],
        'bom': [pp_bom, pn_bom],
        'dist_cls': dist_cls,
        'dist_ave': dist_ave,
        'dist_bom': dist_bom,
    }, './artifact/visualization/fluo_lohi_data.pt')

def plot_fluo_lohi(load=True):
    lohi_data = torch.load('./artifact/visualization/fluo_lohi_data.pt')
    dist_ave, dist_cls, dist_bom = lohi_data['dist_ave'], lohi_data['dist_cls'], lohi_data['dist_bom']
    
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    vmin = min(np.min(dist_bom), np.min(dist_ave), np.min(dist_cls))
    vmax = max(np.max(dist_bom), np.max(dist_ave), np.max(dist_cls))

    im0 = axes[0].imshow(dist_cls, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    im1 = axes[1].imshow(dist_ave, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    im2 = axes[2].imshow(dist_bom, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    xticks, yticks = np.array([4, 9, 14, 19]), np.array([4, 9, 14, 19])
    xlabel, ylabel = xticks + 1, yticks + 1
    axes[0].set_xlabel('CLS-Pooling\nDistance', fontsize=10)
    axes[0].set_xticks(xticks, labels=xlabel)
    axes[0].set_yticks(yticks, labels=ylabel)
    axes[1].set_xlabel('Avg-Pooling\nDistance', fontsize=10)
    axes[1].set_xticks(xticks, labels=xlabel)
    axes[1].set_yticks(yticks, labels=ylabel)
    axes[2].set_xlabel('Minimum Local\nAvg-Pooling Distance', fontsize=10)
    axes[2].set_xticks(xticks, labels=xlabel)
    axes[2].set_yticks(yticks, labels=ylabel)

    fig.subplots_adjust(left=0.15, right=0.95, wspace=0.3, bottom=0.25, top=0.95)
    cbar_ax = fig.add_axes([0.02, 0.25, 0.02, 0.67])
    fig.colorbar(im0, cax=cbar_ax)
    plt.savefig('./artifact/visualization/fluo_lohi_activity_heatmap.png')

def plot_fig_2_full(avg_cosim, cls_cosim):
    xticks, yticks = np.array([4, 9, 14, 19]), np.array([4, 9, 14, 19])
    xlabel, ylabel = xticks + 1, yticks + 1

    lohi_data = torch.load('./artifact/visualization/fluo_lohi_data.pt')
    dist_ave, dist_cls, dist_bom = lohi_data['dist_ave'], lohi_data['dist_cls'], lohi_data['dist_bom']
    

    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.5))

    vmin = min(np.min(avg_cosim), np.min(cls_cosim))
    vmax = max(np.max(avg_cosim), np.max(cls_cosim))
    
    for i in range(5):
        axes[i].set_xticks(xticks, labels=xlabel)
        axes[i].set_yticks(yticks, labels=ylabel)

    im = axes[0].imshow(avg_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].imshow(cls_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    
    axes[0].set_xlabel('Avg-Pooling\nCosine Similarity', fontsize=10)
    axes[1].set_xlabel('CLS-Pooling\nCosine Similarity', fontsize=10)
    
    axes[0].set_title('(a)')
    axes[1].set_title('(b)')

    im = axes[2].imshow(dist_ave, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[3].imshow(dist_cls, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[4].imshow(dist_bom, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')

    axes[2].set_xlabel('Avg-Pooling\nCosine Similarity', fontsize=10)
    axes[3].set_xlabel('CLS-Pooling\nCosine Similarity', fontsize=10)
    axes[4].set_xlabel('Minimum Local Avg-Pooling\n Cosine Similarity', fontsize=10)
    
    axes[2].set_title('(c)')
    axes[3].set_title('(d)')
    axes[4].set_title('(e)')

    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.3, bottom=0.25, top=0.85)
    cbar_ax = fig.add_axes([0.02, 0.25, 0.02, 0.60])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(f'./artifact/visualization/fig2.png')

def plot_fig_2(avg_cosim, cls_cosim):
    xticks, yticks = np.array([4, 9, 14, 19]), np.array([4, 9, 14, 19])
    xlabel, ylabel = xticks + 1, yticks + 1

    lohi_data = torch.load('./artifact/visualization/fluo_lohi_data.pt')
    dist_ave, dist_cls, dist_bom = lohi_data['dist_ave'], lohi_data['dist_cls'], lohi_data['dist_bom']
    

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

    vmin = min(np.min(avg_cosim), np.min(cls_cosim))
    vmax = max(np.max(avg_cosim), np.max(cls_cosim))
    
    for i in range(2):
        axes[i].set_xticks(xticks, labels=xlabel)
        axes[i].set_yticks(yticks, labels=ylabel)

    im = axes[0].imshow(avg_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].imshow(cls_cosim, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    
    axes[0].set_xlabel('Avg-Pooling\nCosine Similarity', fontsize=10)
    axes[1].set_xlabel('CLS-Pooling\nCosine Similarity', fontsize=10)
    
    axes[0].set_title('(a)')
    axes[1].set_title('(b)')

    fig.subplots_adjust(left=0.15, right=0.95, wspace=0.3, bottom=0.25, top=0.85)
    cbar_ax = fig.add_axes([0.02, 0.25, 0.02, 0.60])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(f'./artifact/visualization/fig2a.png')

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    
    vmin = min(np.min(dist_ave), np.min(dist_cls), np.min(dist_bom))
    vmax = max(np.max(dist_ave), np.max(dist_cls), np.max(dist_bom))
    for i in range(3):
        axes[i].set_xticks(xticks, labels=xlabel)
        axes[i].set_yticks(yticks, labels=ylabel)

    im = axes[0].imshow(dist_ave, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].imshow(dist_cls, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')
    axes[2].imshow(dist_bom, cmap='PiYG', vmin=vmin, vmax=vmax, origin='lower')

    axes[0].set_xlabel('Avg-Pooling\nCosine Similarity', fontsize=10)
    axes[1].set_xlabel('CLS-Pooling\nCosine Similarity', fontsize=10)
    axes[2].set_xlabel('Minimum Local Avg-Pooling\n Cosine Similarity', fontsize=10)
    
    axes[0].set_title('(c)')
    axes[1].set_title('(d)')
    axes[2].set_title('(e)')

    fig.subplots_adjust(left=0.15, right=0.95, wspace=0.3, bottom=0.25, top=0.85)
    cbar_ax = fig.add_axes([0.02, 0.25, 0.02, 0.60])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(f'./artifact/visualization/fig2b.png')
    

if __name__ == '__main__':
    retrieve_fluo_lohi(7, 1)
    # exit()

    avg_pooled_samples, cls_pooled_samples, bom_pooled_samples, full_embedding_samples = retrieve_scope_samples()
    avg_cosim, cls_cosim = pairwise_cosim_avg_cls(avg_pooled_samples, cls_pooled_samples)
    bom_cosim = pairwise_cosim_bom(bom_pooled_samples)
    # print(bom_cosim)
    # exit()
    plot_fig_2_full(avg_cosim.numpy(), cls_cosim.numpy())
    plot_fig_2(avg_cosim.numpy(), cls_cosim.numpy())
    # plot_fig_2(avg_cosim.numpy(), cls_cosim.numpy(), bom_cosim.numpy())
    
    # cosim_plot(avg_cosim.numpy(), cls_cosim.numpy())
    # avg_infoloss = infoloss_avg_cls(avg_pooled_samples, full_embedding_samples)
    # cls_infoloss = infoloss_avg_cls(cls_pooled_samples, full_embedding_samples)
    # bom_infoloss = infoloss_bom(bom_pooled_samples, full_embedding_samples)           
    # plot_infoloss_vs_variance(cls_infoloss, avg_infoloss, bom_infoloss, full_embedding_samples)
    # infoloss_plot(avg_infoloss, cls_infoloss, bom_infoloss)