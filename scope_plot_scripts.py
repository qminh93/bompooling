from utils import *
from datasets import *

ARTIFACT_DIR = {
    'scope_a': f'./artifact/exp8',
    'scope_all': f'./artifact/exp9',
    'scope_plots': f'./artifact/scope_plots'
}

pooling_method = {
    f'bom_k100_s80': 'BoM-Pooling', 
    'avg': 'Avg-Pooling',
    'cls': 'CLS-Pooling', 
    'sep': 'EoS-Pooling', 
}
emb_models = {
    'prottrans': 'ProtTrans',
    'protbert': 'ProtBERT',
    'esm2-35M': 'ESM-2 (35M)',
    'esm2-150M': 'ESM-2 (150M)',
    'esm2-650M': 'ESM-2 (650M)'
}
seeds = [
    261,
    # 2602,
    # 26003,
    # 2604,
    # 265
]

def compute_label_mat(cls, remote_only=True):
    data = SCOPev2Dataset(
        emb_model='prottrans',
        save='load_pool',
        pooling_mode='cls',
        k=100, s=80, cls=cls
    )

    test_idx = [i for i in trange(len(data.test_adj_list)) if (len(data.test_adj_list[i]) > 1) and (len(data.test_adj_list[i]) < 200)]
    n_test = len(test_idx)
    label_mat = torch.zeros(n_test, n_test)
    for i in range(n_test):
        sfi, fai, _ = data.test_data['seq_list'][test_idx[i]]
        for j in range(n_test):
            sfj, faj, _ = data.test_data['seq_list'][test_idx[j]]
            if sfi == sfj:
                if remote_only and (fai == faj):
                    label_mat[i, j] = .5
                else:
                    label_mat[i, j] = 1.
    torch.save(label_mat, f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_label_mat{"_remote" if remote_only else ""}.pt')

def compute_roc_prc(cls, remote_only=True):
    label_mat = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_label_mat{"_remote" if remote_only else ""}.pt')
    triu_idx = torch.triu_indices(label_mat.shape[0], label_mat.shape[0], 1)
    label_no_easy = label_mat[torch.where(label_mat != .5)]
    au = {'roc': defaultdict(dict), 'prc': defaultdict(dict)}
    for i, (em, em_label) in enumerate(emb_models.items()):
        for j, (pm, pm_label) in enumerate(pooling_method.items()):
            result = []
            for seed in seeds:
                res = torch.load(f'{ARTIFACT_DIR[f"scope_{cls}"]}/scope_v2_contrastive_{em}_{pm}_seed{seed}.pt')['dist'][-1]
                # res = (inverse_sigmoid(res, 2.5) + 1.) / 2.
                result.append(res)
            result = torch.mean(torch.stack(result, dim=0), dim=0)
            
            result_mat = torch.zeros(label_mat.shape[0], label_mat.shape[0])
            result_mat[triu_idx[0], triu_idx[1]] = result
            result_mat = 1. - (result_mat + result_mat.t())
            result_no_easy = result_mat[torch.where(label_mat != .5)]

            fpr, tpr, roc_threshold = roc_curve(label_no_easy, result_no_easy)
            pre, rec, prc_threshold = precision_recall_curve(label_no_easy, result_no_easy)
            au['roc'][em][pm] = (roc_auc_score(label_no_easy, result_no_easy), fpr, tpr)
            au['prc'][em][pm] = (average_precision_score(label_no_easy, result_no_easy), pre, rec)
    au['roc'] = dict(au['roc'])
    au['prc'] = dict(au['prc'])
    torch.save(au, f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_au.pt')

def compute_perseq_roc_prc(cls, remote_only=True):
    label_mat = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_label_mat{"_remote" if remote_only else ""}.pt')
    n_test = label_mat.shape[0]
    triu_idx = torch.triu_indices(label_mat.shape[0], label_mat.shape[0], 1)
    perseq_au = {'roc': defaultdict(lambda: defaultdict(list)), 'prc': defaultdict(lambda: defaultdict(list))}
    for i, (em, em_label) in enumerate(emb_models.items()):
        for j, (pm, pm_label) in enumerate(pooling_method.items()):
            result = []
            for seed in seeds:
                res = torch.load(f'{ARTIFACT_DIR[f"scope_{cls}"]}/scope_v2_contrastive_{em}_{pm}_seed{seed}.pt')['dist'][-1]
                # res = (inverse_sigmoid(res, 2.5) + 1.) / 2.
                result.append(res)
            result = torch.mean(torch.stack(result, dim=0), dim=0)
            result_mat = torch.zeros(label_mat.shape[0], label_mat.shape[0])
            result_mat[triu_idx[0], triu_idx[1]] = result
            result_mat = 1. - (result_mat + result_mat.t())

            for t in range(n_test):
                remote_idx = torch.where(label_mat[t] == 1)[0]
                if remote_idx.shape[0] == 0:
                    continue 
                exclude_normal_homolog_idx = torch.where(label_mat[t] != .5)
                lt = label_mat[t][exclude_normal_homolog_idx]
                rt = result_mat[t][exclude_normal_homolog_idx]
                fpr, tpr, roc_threshold = roc_curve(lt, rt)
                pre, rec, prc_threshold = precision_recall_curve(lt, rt)
                perseq_au['roc'][em][pm].append((roc_auc_score(lt, rt), fpr, tpr))
                perseq_au['prc'][em][pm].append((average_precision_score(lt, rt), pre, rec))
                
            
        perseq_au['roc'][em] = dict(perseq_au['roc'][em])
        perseq_au['prc'][em] = dict(perseq_au['prc'][em])
    perseq_au['roc'] = dict(perseq_au['roc'])
    perseq_au['prc'] = dict(perseq_au['prc'])
    torch.save(perseq_au, f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_perseq_au.pt')

def compute_ndcg(cls, remote_only=True, topk=None):
    label_mat = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_label_mat{"_remote" if remote_only else ""}.pt')
    n_test = label_mat.shape[0]
    triu_idx = torch.triu_indices(label_mat.shape[0], label_mat.shape[0], 1)
    exclude_normal_homolog = [torch.where(label_mat[t] != 0.5) for t in range(n_test)]
    normalization = torch.tensor([
        np.sum([1./np.log2(i + 2.) for i in range(torch.where(label_mat[t] == 1.)[0].shape[0])])
        for t in range(n_test)
    ])
    has_remote = torch.where(normalization > 0)
    mnrr = defaultdict(dict)
    for i, (em, em_label) in enumerate(emb_models.items()):
        for j, (pm, pm_label) in enumerate(pooling_method.items()):
            result = []
            for seed in seeds:
                result.append(torch.load(f'{ARTIFACT_DIR[f"scope_{cls}"]}/scope_v2_contrastive_{em}_{pm}_seed{seed}.pt')['dist'][-1])
            result = torch.mean(torch.stack(result, dim=0), dim=0)
            result_mat = torch.zeros(n_test, n_test)
            result_mat[triu_idx[0], triu_idx[1]] = result
            result_mat = result_mat + result_mat.t()
            
            mnrr[em][pm] = torch.zeros(n_test)
            for t in range(n_test):
                sorted_indices = torch.argsort(result_mat[t][exclude_normal_homolog[t]], dim=-1)
                n = sorted_indices.shape[0]
                t_rank = torch.zeros(n)
                t_rank[sorted_indices] = torch.arange(n) + 2.
                mnrr[em][pm][t] = (label_mat[t][exclude_normal_homolog[t]] / torch.log2(t_rank)).sum()
            mnrr[em][pm][has_remote] = mnrr[em][pm][has_remote] / normalization[has_remote].float()
    
    torch.save(dict(mnrr), f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_mnrr.pt')

def plot_roc_prc(cls, remote_only=True):
    label_mat = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_label_mat{"_remote" if remote_only else ""}.pt')
    triu_idx = torch.triu_indices(label_mat.shape[0], label_mat.shape[0], 1)
    label_no_easy = label_mat[torch.where(label_mat != .5)]

    fig, axes = plt.subplots(2, 5, figsize=(12.5, 7.))
    au = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_au.pt')
    for i, (em, em_label) in enumerate(emb_models.items()):
        for j, (pm, pm_label) in enumerate(pooling_method.items()):
            auroc, fpr, tpr = au['roc'][em][pm]
            auprc, pre, rec = au['roc'][em][pm]
            axes[0, i].plot(fpr, tpr, label=pm_label)
            axes[1, i].plot(rec, pre, label=pm_label)
        axes[0, i].set_title(em_label)
        axes[0, i].set_xlabel('False Positive Rate')
        axes[0, i].plot([0, 1], [0, 1], color='black', linestyle='--')
        axes[0, i].set_xlim([-0.05, 1.05])
        axes[0, i].set_ylim([-0.05, 1.05])
        if i == 0:
            axes[0, i].set_ylabel('True Positive Rate')

        axes[1, i].set_title(em_label)
        axes[1, i].set_xlabel('Recall')
        axes[1, i].plot([0, 1], [0, 1], color='black', linestyle='--')
        axes[1, i].set_xlim([-0.05, 1.05])
        axes[1, i].set_ylim([-0.05, 1.05])
        if i == 0:
            axes[1, i].set_ylabel('Precision')
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, hspace=0.3, bottom=0.1, top=0.75)
    axes[0, 0].legend(ncol=5, bbox_to_anchor=(0.4, 1.12), loc='lower left')
    fig.savefig(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_roc.png')

def barplot(cls, plot='roc', ylim=None):
    ems = list(emb_models.keys())
    pms = list(pooling_method.keys())
    if plot == 'ndcg':
        mnrr = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_mnrr.pt')
        metric = {
            pm: [np.mean(mnrr[em][pm].numpy()) for em in ems]
            for pm in pms
        }
        ylabel = 'NDCG'
    else:
        au = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_au.pt')
        metric = {
            pm: [au[plot][em][pm][0] for em in ems]
            for pm in pms
        }
        ylabel = 'AUROC' if plot=='roc' else 'AUPRC'   
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
    for i, (attribute, measurement) in enumerate(metric.items()):
        x = np.arange(5) * 5 + i
        ax.bar(x, measurement, 1, label=pooling_method[attribute])

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(np.arange(5) * 5 + 1.5, list(emb_models.values()))
    ax.legend(bbox_to_anchor=(0.18, -0.15), loc='upper left', ncol=2, fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.35, top=0.95)
    fig.savefig(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_barplot_{plot}.png')

def perseq_barplot(cls, plot='roc', ylim=None):
    au = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_perseq_au.pt')
    ems = list(emb_models.keys())
    pms = list(pooling_method.keys())
    au_perseq = {}
    for pm in pms:
        au_perseq[pm] = []
        for em in ems:
            mean_au = np.mean([t[0] for t in au[plot][em][pm]])
            au_perseq[pm].append(mean_au)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
    for i, (attribute, measurement) in enumerate(au_perseq.items()):
        x = np.arange(5) * 5 + i
        ax.bar(x, measurement, 1, label=pooling_method[attribute])

    ax.set_ylim(ylim)
    ax.set_ylabel('Mean AUROC' if plot=='roc' else 'Mean AUPRC', fontsize=12)
    ax.set_xticks(np.arange(5) * 5 + 1.5, list(emb_models.values()))
    ax.legend(bbox_to_anchor=(0.18, -0.15), loc='upper left', ncol=2, fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.35, top=0.95)

    fig.savefig(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_barplot_perseq_{plot}.png')

def mnrr_barplot(cls, plot='roc', mnrr_ylim=None):
    mnrr = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_mnrr.pt')
    au = torch.load(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_au.pt')
    ems = list(emb_models.keys())
    pms = list(pooling_method.keys())
    mnrr = {
        pm: [np.mean(mnrr[em][pm].numpy()) for em in ems]
        for pm in pms
    }
    au = {
        pm: [au[plot][em][pm] for em in ems]
        for pm in pms
    }
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3))
    for i, (attribute, measurement) in enumerate(mnrr.items()):
        x = np.arange(5) * 5 + i
        print(attribute, measurement)
        ax[0].bar(x, measurement, 1, label=pooling_method[attribute])

    ax[0].set_ylim(mnrr_ylim)
    ax[0].set_ylabel('MNRR', fontsize=12)
    ax[0].set_xticks(np.arange(5) * 5 + 1.5, list(emb_models.values()))
    ax[0].legend(bbox_to_anchor=(0.38, -0.3), loc='upper left', ncol=4, fontsize=12)

    for i, (attribute, measurement) in enumerate(au.items()):
        x = np.arange(5) * 5 + i
        # print(attribute, measurement)
        ax[1].bar(x, measurement, 1, label=pooling_method[attribute])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax[1].set_ylim([.4,.9])
    ax[1].set_ylabel('AUROC' if plot=='roc' else 'AUPRC', fontsize=12)
    ax[1].set_xticks(np.arange(5) * 5 + 1.5, list(emb_models.values()))
    fig.subplots_adjust(left=0.07, right=0.99, wspace=0.15, bottom=0.35, top=0.95)
    fig.savefig(f'{ARTIFACT_DIR["scope_plots"]}/scope_{cls}_barplot_mnrr_{plot}.png')


if __name__ == '__main__':
    # compute_label_mat('a')
    # compute_label_mat('all')
    # compute_roc_prc('a')
    # compute_roc_prc('all')
    # compute_perseq_roc_prc('a')
    # compute_perseq_roc_prc('all')
    barplot('all', 'roc', ylim=[0.5, 0.85])
    # barplot('all', 'prc', ylim=[0.15, 0.3])
    barplot('all', 'ndcg', ylim=[0.5, 0.62])
    barplot('a', 'roc', ylim=[0.5, 0.85])
    # barplot('a', 'prc', ylim=[0.25, 0.4])
    barplot('a', 'ndcg', ylim=[0.5, 0.7])
    # perseq_barplot('all', 'roc', ylim=[0.5, 0.8])
    # perseq_barplot('all', 'prc', ylim=[0.2, 0.35])
    # perseq_roc_prc_barplot('a', 'roc', ylim=[0.5, 0.8])
    # perseq_roc_prc_barplot('a', 'prc', ylim=[0.36, 0.45])
    # plot_roc_prc('a')
    # plot_roc_prc('all')
    