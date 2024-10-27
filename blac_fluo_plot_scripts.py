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

def latex_table(mean_table, std_table, column_headers, row_headers, output_file):
    fout = open(output_file, 'w')
    r, c = std_table.shape[0], mean_table.shape[1]
    assert r == len(row_headers) and c == len(column_headers), "Mismatched no. headers"
    
    all_row_latex_code = []

    header_row_latex_code = '& ' + '& '.join(column_headers) + '\\\\'
    all_row_latex_code.append('\\toprule')
    all_row_latex_code.append(header_row_latex_code)
    all_row_latex_code.append('\\midrule')
    for i in range(r):
        row_latex_code = row_headers[i]
        for j in range(c):
            row_latex_code += f'& ${mean_table[i, j]:.3f} ({std_table[i, j]:.3f})$ '
        all_row_latex_code.append(row_latex_code + '\\\\')
    all_row_latex_code.append('\\bottomrule')
    for line in all_row_latex_code:
        fout.write(line + '\n')

def generate_regression_table(task, k, s):
    pooling_method = [
        # 'cls', 'sep', 'avg', 
        'vanilla_cls', 'vanilla_sep', 'vanilla_avg', 
        f'bom_k{k}_s{s}'
    ]
    emb_models = [
        'prottrans', 
        'protbert',
        'esm2-35M', 
        'esm2-650M',
        'esm2-150M'    
    ]
    seeds = [
        261,
        2602,
        26003,
        2604,
        265
    ]

    mean_table, std_table = np.zeros((len(pooling_method), len(emb_models))), np.zeros((len(pooling_method), len(emb_models)))
    row_headers = [
        # '\\textsc{Cls-Pooling}',
        # '\\textsc{EoS-Pooling}',
        # '\\textsc{Avg-Pooling}',
        '\\textsc{V-Cls-Pooling}',
        '\\textsc{V-EoS-Pooling}',
        '\\textsc{V-Avg-Pooling}',
        '\\textsc{BoM-Pooling}'
    ]
    col_headers = [
        'ProtTrans',
        'ProtBERT',
        'ESM-2 (35M)',
        'ESM-2 (150M)',
        'ESM-2 (650M)'
    ]
    for i, em in enumerate(emb_models):           
        for j, pm in enumerate(pooling_method):
            error = []
            for seed in seeds:
                res = torch.load(f'{ARTIFACT_DIR[task]}/{task}_regression_{em}_{pm}_seed{seed}.pt')
                error.append(res['error'][-1])           
            mean_table[j, i] = np.mean(error)
            std_table[j, i] = np.std(error)

    latex_table(mean_table, std_table, col_headers, row_headers, f'{ARTIFACT_DIR[task]}/{task}_regression_table.txt')    
    
def generate_preference_table(task, k, s):
    pooling_method = [
        # 'cls', 'sep', 'avg', 
        'vanilla_cls', 'vanilla_sep', 'vanilla_avg', 
        f'bom_k{k}_s{s}'
    ]
    emb_models = ['prottrans', 'protbert', 'esm2-35M', 'esm2-650M', 'esm2-150M']
    seeds = [
        261,
        2602,
        26003,
        2604,
        265
    ]

    mean_table, std_table = np.zeros((len(pooling_method), len(emb_models))), np.zeros((len(pooling_method), len(emb_models)))
    row_headers = [
        # '\\textsc{Cls-Pooling}',
        # '\\textsc{EoS-Pooling}',
        # '\\textsc{Avg-Pooling}',
        '\\textsc{V-Cls-Pooling}',
        '\\textsc{V-EoS-Pooling}',
        '\\textsc{V-Avg-Pooling}',
        '\\textsc{BoM-Pooling}'
    ]
    col_headers = [
        'ProtTrans',
        'ProtBERT',
        'ESM-2 (35M)',
        'ESM-2 (150M)',
        'ESM-2 (650M)'
    ]

    for u, em in enumerate(emb_models):
        test_set = torch.load(f'{DATA_DIR[task]}/{task}_preference_test1.pt')
        data_type = TapeRegressionDataset if task in ['fluorescence', 'stability'] else PEERRegressionDataset
        data = data_type(
            dataset=task,
            emb_model=em,
            save='load_pool',
            pooling_mode='cls',
            k=k, stride=s
        )
        y_test = np.array([
            (data.y_test[test_set['left'][i]] > data.y_test[test_set['right'][i]]).int()
            for i in range(data.y_test.shape[0])
        ])
        for v, pm in enumerate(pooling_method):    
            auc = []
            for seed in seeds:
                result = torch.load(f'{ARTIFACT_DIR[task]}/{task}_preference_{em}_{pm}_seed{seed}.pt')
                prob = torch.cat(result['prob'][-1][0]).cpu().numpy()
                auc.append(roc_auc_score(y_test, prob))
            mean_table[v, u], std_table[v, u] = np.mean(auc), np.std(auc)
    
    latex_table(mean_table, std_table, col_headers, row_headers, f'{ARTIFACT_DIR[task]}/{task}_preference_table.txt')    


def plot_preference(task, k, s):
    pooling_method = {
        f'bom_k{k}_s{s}': 'BoM-Pooling',
        'cls': 'T-CLS-Pooling', 
        'sep': 'T-EoS-Pooling', 
        'ave': 'T-Avg-Pooling'
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
        2602,
        26003,
        2604,
        265
    ]
    
    fig, axes = plt.subplots(1, len(emb_models.keys()), figsize=(12, 3.5), sharey=True)
    for t, (em, em_label) in enumerate(emb_models.items()):
        test_set = torch.load(f'{DATA_DIR[task]}/{task}_preference_test1.pt')
        data_type = TapeRegressionDataset if task in ['fluorescence', 'stability'] else PEERRegressionDataset
        data = data_type(
            dataset=task,
            emb_model=em,
            save='load_pool',
            pooling_mode='cls',
            k=k, stride=s
        )
        y_test = np.array([
            (data.y_test[test_set['left'][i]] > data.y_test[test_set['right'][i]]).int()
            for i in range(data.y_test.shape[0])
        ])
        for pm, pm_label in pooling_method.items():    
            prob = []
            for seed in seeds:
                result = torch.load(f'{ARTIFACT_DIR[task]}/{task}_preference_{em}_{pm}_seed{seed}.pt')
                prob.append(torch.cat(result['prob'][-1][0]).to('cpu'))
                
            prob = torch.mean(torch.stack(prob, dim=0), dim=0)
            prob = prob.cpu().numpy()
            fpr, tpr, _ = roc_curve(y_test, prob)
            axes[t].plot(fpr, tpr, label=pm_label)

        for pm, pm_label in pooling_method.items():    
            if pm == f'bom_k{k}_s{s}':
                continue
            prob = []
            for seed in seeds:
                result = torch.load(f'{ARTIFACT_DIR[task]}/{task}_preference_{em}_vanilla_{pm}_seed{seed}.pt')
                prob.append(torch.cat(result['prob'][-1][0]).to('cpu'))
            prob = torch.mean(torch.stack(prob, dim=0), dim=0)
            prob = prob.cpu().numpy()
            fpr, tpr, _ = roc_curve(y_test, prob)
            axes[t].plot(fpr, tpr, label=pm_label[2:])

        axes[t].plot([0, 1], [0, 1], color='black', linestyle='--')
        axes[t].set_xlim([0.0, 1.0])
        axes[t].set_ylim([0.0, 1.05])
        axes[t].set_xlabel('False Positive Rate')
        if t == 0:
            axes[t].set_ylabel('True Positive Rate')
        axes[t].set_title(em_label)
        
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.2, bottom=0.2, top=0.8)
    axes[0].legend(ncol=7, bbox_to_anchor=(-0.14, 1.12), loc='lower left')
    fig.savefig(f'{ARTIFACT_DIR[task]}/{task}_preference_roc.png')
        
    
if __name__ == '__main__':
    generate_regression_table('fluorescence', 100, 20)
    generate_preference_table('beta_lactamase', 100, 20)