from models import *
from nets import *
from contrastive import *

ARTIFACT_DIR = {
    'dpi': f'./artifact/exp1_time',
}

def run_contrastive(**kwargs):
    set_seed(kwargs["seed"])
    save_dir = f'{ARTIFACT_DIR[kwargs["dataset"]]}/{kwargs["dataset"]}_contrastive_{kwargs["emb_model"]}_{kwargs["pooling_mode"]}'
    if kwargs['pooling_mode'] == 'bom':
        save_dir += f'_k{kwargs["k"]}_s{kwargs["stride"]}' 
    save_dir += f'_seed{kwargs["seed"]}.pt'
    
    print(f'Running {kwargs["dataset"]} contrastive with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
    dist_net_config = {
        'qnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'knet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'vnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024])
    }
    dist_net = CrossAttentionKernel(**dist_net_config) if kwargs['pooling_mode'] == 'bom' else MultiheadLinearKernel(**dist_net_config) 
    if kwargs['dataset'] == 'dpi':
        data = DPIDataset(
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
        model = DPI(
            data=data,
            contrastive_loss=TripletLoss(dist_net, margin=kwargs.get('margin', 1.)),
        )
    model.reset_history()
    model.train(
        n_epochs=kwargs.get('n_epochs', 401),
        interval=kwargs.get('interval', 20),
        batch_size=kwargs.get('batch_size', 2),
        lr=kwargs.get('lr', 1e-4)
    )
    if kwargs['save']:
        torch.save(model.history, save_dir)

    return model

def plot_time():
    exp_folder = f'./artifact/exp1_time'

    methods = {
        f'bom_k{max(1, 20*i)}_s{max(1, 4*i)}': f'BoM-Pooling (k={max(1, 20*i)})'
        for i in range(6)
    } 
    methods['avg'] = 'Avg-Pooling'
    plt.figure(figsize=(5, 4.5))
    for m, m_label in methods.items():
        time = torch.load(f'{exp_folder}/dpi_contrastive_esm2-650M_{m}_seed261.pt')['training_time']
        for j in range(1, len(time)):
            time[j] += time[j-1]
        # scale by 700 because there are roughly 700 peptides for 1 domains
        # in practice 1 epoch cycles through all domains
        # but we want to measure time for 1 epoch that cycles through all data (all method x 700 so it's fair comparison)
        plt.plot(np.arange(len(time)), np.array(time) * 700, '-x', markevery=10, label=m_label)
    plt.xlabel('Training Epochs')
    plt.ylabel('Training Time (s)')
    plt.legend(ncol=2, bbox_to_anchor=(1.02, -0.2))
    plt.subplots_adjust(bottom=0.35, left=0.15, right=0.95, top=0.97)

    plt.savefig(f'{exp_folder}/time_compare_k.png')

def run_experiments():
    torch.cuda.set_device(1)
    em = 'esm2-650M'
    common_kwargs = {
        'seed': 2602,
        'dataset': 'dpi',
        'save': True,
        'n_epochs': 101,
        'interval': 101,
        'margin': 0.6,
        'list_only': False
    }
    run_contrastive(**common_kwargs, pooling_mode='avg', emb_model=em, lr=1e-4) 
    ks = [
        1,
        20, 
        40, 
        60, 
        80, 
        100
    ]
    for k in ks:
        common_kwargs = {
            'seed': 261,
            'dataset': 'dpi',
            'save': True,
            'k': k,
            'stride': max(k//5, 1),
            'n_epochs': 101,
            'interval': 101,
            'margin': 0.6,
            'list_only': False
        }
        run_contrastive(**common_kwargs, pooling_mode=f'bom', emb_model=em, lr=1e-4)

if __name__ == '__main__':
    run_experiments()
    plot_time()