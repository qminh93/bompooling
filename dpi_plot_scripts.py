from utils import *
from datasets import *
ARTIFACT_DIR = {
    'fluorescence': f'./artifact/exp3',
    'stability': f'./artifact/exp4',
    'beta_lactamase': f'./artifact/exp6',
    'solubility': f'./artifact/exp7',
    'dpi': f'./artifact/exp1',
    'ppi': f'./artifact/exp2_v3',
    'scope': f'./artifact/exp5',
    'scope_v2': f'./artifact/exp8',
}

def plot_dpi(k, s):
    pooling_methods = {
        f'bom_k{k}_s{s}': 'BoM', 
        'avg': 'Avg'
    }
    test_domains = ['PTB', 'PTP', 'Kinase_TK']
    emb_models = {
        'prottrans': 'ProtTrans',
        'esm2-150M': 'ESM-2 (150M)',
        'esm2-650M': 'ESM-2 (650M)'
    }
    seeds = [
        261,
        2602,
        26003,
        2604,
        265
    ]
    
    # Netphorest
    netphorest_label, netphorest_pred = [], []
    for dom in test_domains:
        netphorest_result = open(f'{DATA_ROOT}/protein-peptide/baselines/netphorest/{dom}/{dom}_netphorest_predictions.csv')
        lines = netphorest_result.readlines()[1:]
        for l in lines:
            tokens = l.split(',')
            netphorest_label.append(int(tokens[-2]))
            netphorest_pred.append(float(tokens[-1]))
    netphorest_label, netphorest_pred = np.array(netphorest_label), np.array(netphorest_pred)
    netphorest_fpr, netphorest_tpr, _ = roc_curve(netphorest_label, netphorest_pred)
    netphorest_auc = roc_auc_score(netphorest_label, netphorest_pred)
    
    # PSSM 
    pssm_label, pssm_pred = [], []
    for dom in test_domains:
        pssm_result = open(f'{DATA_ROOT}/protein-peptide/baselines/pssm/output/{dom}_pssm_predictions.csv')
        lines = pssm_result.readlines()[1:]
        for l in lines:
            tokens = l.split(',')
            pssm_label.append(int(tokens[4]))
            pssm_pred.append(float(tokens[-1]))
    pssm_label, pssm_pred = np.array(pssm_label), np.array(pssm_pred)
    pssm_fpr, pssm_tpr, _ = roc_curve(pssm_label, pssm_pred)
    pssm_auc = roc_auc_score(pssm_label, pssm_pred)

    # PLMs
    fig, axes = plt.subplots(1, 3, figsize=(6, 3.5))
    for t, (em, em_label) in enumerate(emb_models.items()):
        labels = []
        for dom in test_domains:
            data = DPIDataset(
                emb_model=em,
                save='load_pool',
                pooling_mode='bom',
                k=k, stride=s
            ).test_data
            pos, neg = data[dom]['pos'], data[dom]['neg']
            for i in range(len(pos)):
                labels += [1. for _ in pos[i]]
            for i in range(len(neg)):
                labels += [0. for _ in neg[i]]
        # labels combining 3 test domains
        labels = np.array(labels)

        for pm, pm_label in pooling_methods.items():
            dist = []
            for dom in test_domains:
                dom_dist = []
                for seed in seeds:
                    result = torch.load(f'{ARTIFACT_DIR["dpi"]}/dpi_contrastive_{em}_{pm}_seed{seed}.pt')
                    dom_dist.append(1. - torch.tensor(result[dom]['dist'][-1]))
                dist.append(torch.mean(torch.stack(dom_dist, dim=0), dim=0))
            dist = torch.cat(dist, dim=0).cpu().numpy()
            fpr, tpr, _ = roc_curve(labels, dist)
            auc = roc_auc_score(labels, dist)
            axes[t].plot(fpr, tpr, label=f'{em_label} [{pm_label}] ({auc:.3f})')
        
        axes[t].plot(netphorest_fpr, netphorest_tpr, label=f'NetPhorest ({netphorest_auc:.3f})')
        axes[t].plot(pssm_fpr, pssm_tpr, label=f'PSSM ({pssm_auc:.3f})')
        axes[t].plot([0, 1], [0, 1], color='black', linestyle='--')
        axes[t].set_xlabel(f'False Positive Rate')
        if t == 0:
            axes[t].set_ylabel(f'True Positive Rate')
        axes[t].legend()
        axes[t].set_xlim([0.5, 1.0])
        axes[t].set_ylim([0.5, 1.0])

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.99, wspace=0.15)
    fig.savefig(f'{ARTIFACT_DIR["dpi"]}/dpi_contrastive_roc.png')

def plot_dpi_bar(k, s):
    pooling_methods = {
        'avg': 'Avg',
        f'bom_k{k}_s{s}': f'BoM (k={k})', 
    }
    test_domains = ['SH2']
    emb_models = {
        'esm2-150M': 'ESM-2 (150M)',
        'esm2-650M': 'ESM-2 (650M)',
        'prottrans': 'ProtTrans'
    }
    seeds = [
        261,
        # 2602,
        # 26003,
        # 2604,
        # 265
    ]
    auc = []
    methods = [
        'PSSM', 'NetPhorest', 'PepInt',
        'ESM-2 (150M) [Avg]', f'ESM-2 (150M) [BoM, k={k}]', 
        'ESM-2 (650M) [Avg]', f'ESM-2 (650M) [BoM, k={k}]', 
        'ProtTrans [Avg]', f'ProtTrans [BoM, k={k}]', 
    ]
    colors = [
        'blue', 'blue', 'blue',
        'green', 'red',
        'green', 'red',
        'green', 'red',
    ]
    
    # PSSM 
    pssm_label, pssm_pred = [], []
    for dom in test_domains:
        pssm_result = open(f'/home/quanghoang_l/data/protein-peptide/baselines/pssm/output/{dom}_pssm_predictions.csv')
        lines = pssm_result.readlines()[1:]
        for l in lines:
            tokens = l.split(',')
            pssm_label.append(int(tokens[4]))
            pssm_pred.append(float(tokens[-1]))
    auc.append(roc_auc_score(pssm_label, pssm_pred))

    # Netphorest
    netphorest_label, netphorest_pred = [], []
    for dom in test_domains:
        netphorest_result = open(f'/home/quanghoang_l/data/protein-peptide/baselines/netphorest/{dom}/{dom}_netphorest_predictions.csv')
        lines = netphorest_result.readlines()[1:]
        for l in lines:
            tokens = l.split(',')
            netphorest_label.append(int(tokens[-2]))
            netphorest_pred.append(float(tokens[-1]))
    auc.append(roc_auc_score(netphorest_label, netphorest_pred))

    # PLMs
    plt.figure(figsize=(4, 4.5))
    for t, (em, em_label) in enumerate(emb_models.items()):
        labels = []
        for dom in test_domains:
            data = DPIDataset(
                emb_model=em,
                save='load_pool',
                pooling_mode='bom',
                k=k, stride=s
            ).test_data
            pos, neg = data[dom]['pos'], data[dom]['neg']
            for i in range(len(pos)):
                labels += [1. for _ in pos[i]]
            for i in range(len(neg)):
                labels += [0. for _ in neg[i]]
        # labels combining 3 test domains
        labels = np.array(labels)

        for pm, pm_label in pooling_methods.items():
            dist = []
            for dom in test_domains:
                dom_dist = []
                for seed in seeds:
                    result = torch.load(f'{ARTIFACT_DIR["dpi"]}/dpi_contrastive_{em}_{pm}_seed{seed}.pt')
                    dom_dist.append(torch.tensor(result[dom]['dist'][-1]))
                dist.append(1. - torch.mean(torch.stack(dom_dist, dim=0), dim=0))
            dist = torch.cat(dist, dim=0).cpu().numpy()
            auc.append(roc_auc_score(labels, dist))

    plt.bar(np.arange(len(methods)), auc, color=colors)
    plt.xticks(np.arange(len(methods)), methods, rotation=45, ha='right')
    plt.ylim([0.6, 0.85])
    plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
    plt.ylabel('AUROC')
    plt.subplots_adjust(bottom=0.35, left=0.17, right=0.97, top=0.97)
    plt.savefig(f'{ARTIFACT_DIR["dpi"]}/dpi_contrastive_auroc_bar.png')

def plot_dpi_compare_k():
    kvals = [1, 20, 40, 60, 80, 100]
    pooling_methods = ['avg'] + [f'bom_k{k}_s{max(1, k//5)}' for k in kvals] 
    test_domains = ['PTB', 'PTP', 'Kinase_TK']
    emb_models = {
        'esm2-650M': 'ESM-2 (650M)',
        'prottrans': 'ProtTrans'
    }
    seeds = [
        261,
        2602,
        26003,
        2604,
        # 265
    ]
    methods = [
        'ESM-2 (650M) [Avg]', 
        'ESM-2 (650M) [BoM, k=1]', 
        'ESM-2 (650M) [BoM, k=20]', 
        'ESM-2 (650M) [BoM, k=40]', 
        'ESM-2 (650M) [BoM, k=60]', 
        'ESM-2 (650M) [BoM, k=80]', 
        'ESM-2 (650M) [BoM, k=100]',
        'ProtTrans [Avg]', 
        'ProtTrans [BoM, k=1]',
        'ProtTrans [BoM, k=20]', 
        'ProtTrans [BoM, k=40]', 
        'ProtTrans [BoM, k=60]', 
        'ProtTrans [BoM, k=80]', 
        'ProtTrans [BoM, k=100]'
    ]
    color = ['#008080', '#FF6F61']
    # PLMs
    plt.figure(figsize=(6, 4.5))
    for t, (em, em_label) in enumerate(emb_models.items()):
        auc = []
        for i in range(7):
            if i == 0:
                pm, k, s = 'avg', 40, 8
            else:
                k, s = max(1, 20 * (i - 1)), max(1, 4 * (i - 1))
                pm = f'bom_k{k}_s{s}'
            labels = []
            for dom in test_domains:
                if k == 1:
                    data = DPIDataset(
                    emb_model=em,
                    save='load_pool',
                    pooling_mode=pm,
                    k=1, stride=1
                ).test_data
                else:
                    data = DPIDataset(
                        emb_model='esm2-150M',
                        save='load_pool',
                        pooling_mode='bom',
                        k=40, stride=8
                    ).test_data
                pos, neg = data[dom]['pos'], data[dom]['neg']
                for j in range(len(pos)):
                    labels += [1. for _ in pos[j]]
                for j in range(len(neg)):
                    labels += [0. for _ in neg[j]]
            # labels combining 3 test domains
            labels = np.array(labels)

            dist = []
            for dom in test_domains:
                dom_dist = []
                for seed in seeds:
                    try:
                        result = torch.load(f'{ARTIFACT_DIR["dpi"]}/dpi_contrastive_{em}_{pm}_seed{seed}.pt')
                        dom_dist.append(1. - torch.tensor(result[dom]['dist'][-1]))
                    except Exception as e:
                        continue
                dist.append(torch.mean(torch.stack(dom_dist, dim=0), dim=0))
            dist = torch.cat(dist, dim=0).cpu().numpy()
            auc.append(roc_auc_score(labels, dist))
            
        plt.bar(np.arange(7) + t * 7, auc, color = color[t], label=em_label)
    plt.xticks(np.arange(len(methods)), methods, rotation=45, ha='right')
    plt.ylim([0.75, 0.85])
    plt.yticks([0.75, 0.8, 0.85])
    plt.ylabel('AUROC')
    plt.legend(loc='upper left')
    plt.subplots_adjust(bottom=0.35, left=0.17, right=0.95, top=0.97)
    plt.savefig(f'{ARTIFACT_DIR["dpi"]}/dpi_contrastive_auroc_compare_k.png')

if __name__ == '__main__':
    plot_dpi_compare_k()
    plot_dpi_bar(40, 8)