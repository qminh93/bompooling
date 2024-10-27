from models import *
from nets import *
from contrastive import *

ARTIFACT_DIR = {
    'fluorescence': f'./artifact/exp3',
    'stability': f'./artifact/exp4',
    'beta_lactamase': f'./artifact/exp6',
    'solubility': f'./artifact/exp7',
    'dpi': f'./artifact/exp1',
    'ppi': f'./artifact/exp2',
    'scope': f'./artifact/exp5',
    'scope_v2': f'./artifact/exp9',
}

for dir in ARTIFACT_DIR.values(): 
    os.makedirs(dir, exist_ok=True)

def run_regression(**kwargs):
    set_seed(kwargs.get('seed'))
    save_dir = f'{ARTIFACT_DIR[kwargs["dataset"]]}/{kwargs["dataset"]}_regression_{kwargs["emb_model"]}_{"" if kwargs["trainable_pool"] else "vanilla_"}{kwargs["pooling_mode"]}'
    if kwargs['pooling_mode'] == 'bom':
        save_dir += f'_k{kwargs["k"]}_s{kwargs["stride"]}' 
    save_dir += f'_seed{kwargs["seed"]}.pt'
    if os.path.isfile(save_dir):
        # print(f'Skipping {kwargs["dataset"]} regression with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
        return
    else:
        print(f'Running {kwargs["dataset"]} regression with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
        if kwargs.get('list_only', False):
            return
    

    if kwargs['dataset'] in ['fluorescence', 'stability']:
        data = TapeRegressionDataset(
            dataset=kwargs['dataset'],
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
    else:
        data = PEERRegressionDataset(
            dataset=kwargs['dataset'],
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        )
    pool_net_config = {
        'qnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'knet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'vnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024])
    }
    if kwargs['pooling_mode'] == 'bom':
        pool_net = SelfAttentionEmbedding(**pool_net_config)
        pred_net=MLP([1024, 256, 32, 1])
    elif kwargs['trainable_pool'] == True:
        pool_net = MultiheadLinear(**pool_net_config)
        pred_net=MLP([1024, 256, 32, 1])
    else:
        pool_net = IdentityEmbedding()
        pred_net=MLP([PLM_dim[kwargs['emb_model']], 256, 32, 1])

    model = RegressionModel(
        data=data,
        pool_net=pool_net.to('cuda'),
        pred_net=pred_net.to('cuda'),
        batch_size=kwargs.get('batch_size', 128)
    )
    model.reset_history()
    model.train(
        n_epochs=kwargs.get('n_epochs', 401),
        interval=kwargs.get('interval', 20),
        lr=kwargs.get('lr', 4e-4)
    )
    if kwargs['save']:
        torch.save(model.history, save_dir)
    return model


def run_preference(**kwargs):
    set_seed(kwargs["seed"])
    save_dir = f'{ARTIFACT_DIR[kwargs["dataset"]]}/{kwargs["dataset"]}_preference_{kwargs["emb_model"]}_{"" if kwargs["trainable_pool"] else "vanilla_"}{kwargs["pooling_mode"]}'
    if kwargs['pooling_mode'] == 'bom':
        save_dir += f'_k{kwargs["k"]}_s{kwargs["stride"]}' 
    save_dir += f'_seed{kwargs["seed"]}.pt'
    if os.path.isfile(save_dir):
        # print(f'Skipping {kwargs["dataset"]} preference with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
        return
    else:
        print(f'Running {kwargs["dataset"]} preference with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
        if kwargs.get('list_only', False):
            return
    
    if kwargs['dataset'] in ['fluorescence', 'stability']:
        data = TapeRegressionDataset(
            dataset=kwargs['dataset'],
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
    else:
        data = PEERRegressionDataset(
            dataset=kwargs['dataset'],
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        )

    pool_net_config = {
        'qnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'knet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'vnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024])
    }
    if kwargs['pooling_mode'] == 'bom':
        pool_net = CrossAttentionPreference(**pool_net_config)   
        rank_net=MLP([1024, 256, 32, 1])
    elif kwargs['trainable_pool']:
        pool_net = MultiheadPreference(**pool_net_config) 
        rank_net=MLP([1024, 256, 32, 1])
    else:
        pool_net = IdentityPreference()
        rank_net=MLP([PLM_dim[kwargs['emb_model']], 256, 32, 1])

    test_sets = [f'{DATA_DIR[kwargs["dataset"]]}/{kwargs["dataset"]}_preference_test1.pt']
    model = PreferenceModel(
        data=data,
        pool_net=pool_net.to('cuda'),
        rank_net=rank_net.to('cuda'),
        batch_size=kwargs.get('batch_size', 128),
        test_sets=test_sets
    )
    model.reset_history()
    model.train(
        n_epochs=kwargs.get('n_epochs', 401),
        interval=kwargs.get('interval', 20),
        lr=kwargs.get('lr', 4e-4)
    )
    if kwargs['save']:
        torch.save(model.history, save_dir)

    return model

def run_contrastive(**kwargs):
    set_seed(kwargs["seed"])
    save_dir = f'{ARTIFACT_DIR[kwargs["dataset"]]}/{kwargs["dataset"]}_contrastive_{kwargs["emb_model"]}_{kwargs["pooling_mode"]}'
    if kwargs['pooling_mode'] == 'bom':
        save_dir += f'_k{kwargs["k"]}_s{kwargs["stride"]}' 
    save_dir += f'_seed{kwargs["seed"]}.pt'
    if os.path.isfile(save_dir):
        # print(f'Skipping {kwargs["dataset"]} contrastive with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
        return
    else:
        print(f'Running {kwargs["dataset"]} contrastive with {kwargs["pooling_mode"]} pooling and {kwargs["emb_model"]} embedding with seed {kwargs["seed"]}')
        if kwargs.get('list_only', False):
            return
    
    dist_net_config = {
        'qnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'knet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024]),
        'vnet': MLP([PLM_dim[kwargs['emb_model']], 256, 1024])
    }
    dist_net = CrossAttentionKernel(**dist_net_config) if kwargs['pooling_mode'] == 'bom' else MultiheadLinearKernel(**dist_net_config) 

    if kwargs['dataset'] == 'ppi':
        data = PPIDataset(
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
        model = PPI(
            data=data,
            contrastive_loss=TripletLoss(dist_net, margin=kwargs.get('margin', 1.)),
        )
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
    if kwargs['dataset'] == 'scope':
        data = SCOPeDataset(
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None)
        ) 
        model = SCOPe(
            data=data,
            contrastive_loss=TripletLoss(dist_net, margin=kwargs.get('margin', 1.)).to('cuda'),
        )
    if kwargs['dataset'] == 'scope_v2':
        data = SCOPev2Dataset(
            emb_model=kwargs['emb_model'],
            save='load_pool',
            pooling_mode=kwargs['pooling_mode'],
            k=kwargs.get('k', None),
            stride=kwargs.get('stride', None),
            cls=kwargs["cls"]
        ) 
        model = SCOPev2(
            data=data,
            contrastive_loss=TripletLoss(dist_net, margin=kwargs.get('margin', 1.)).to('cuda'),
        )

    model.reset_history()
    model.train(
        n_epochs=kwargs.get('n_epochs', 401),
        interval=kwargs.get('interval', 20),
        batch_size=kwargs.get('batch_size', 128),
        lr=kwargs.get('lr', 1e-4)
    )
    if kwargs['save']:
        torch.save(model.history, save_dir)

    return model

def fluorescence_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'fluorescence',
            'save': True,
            'k': 100,
            'stride': 20,
            'n_epochs': 201,
            'interval': 200,
            'list_only': list_only,
            'trainable_pool': False
        },
        lr={
            'regression': 3e-5, 
            # 'preference': 1e-4
        },
    )

def stability_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'stability',
            'save': True,
            'k': 40,
            'stride': 8,
            'n_epochs': 201,
            'interval': 200,
            'list_only': list_only,
            'trainable_pool': True
        },
        lr={
            # 'regression': 1e-4, 
            'preference': 1e-4
        },
    )

def beta_lactamase_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'beta_lactamase',
            'save': True,
            'k': 100,
            'stride': 20,
            'n_epochs': 201,
            'interval': 200,
            'list_only': list_only,
            'trainable_pool': False
        },
        lr={
            # 'regression': 3e-5, 
            'preference': 3e-5
        },
    )

def solubility_experiments(dev,  list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'solubility',
            'save': True,
            'k': 40,
            'stride': 8,
            'n_epochs': 201,
            'interval': 200,
            'list_only': list_only,
            'trainable_pool': True
        },
        lr={
            # 'regression': 3e-5, 
            'preference': 3e-5
        },
    )

def ppi_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'ppi',
            'save': True,
            'k': 100,
            'stride': 20,
            'n_epochs': 201,
            'interval': 200,
            'margin': 0.6,
            'batch_size': 64,
            'list_only': list_only
        },
        lr={'contrastive': 2e-4},
    )

def dpi_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'dpi',
            'save': True,
            'k': 1,
            'stride': 1,
            'n_epochs': 2001,
            'interval': 2000,
            'margin': 0.6,
            'list_only': list_only
        },
        lr={'contrastive': 1e-4},
    )

def scope_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'scope',
            'save': True,
            'k': 100,
            'stride': 20,
            'n_epochs': 201,
            'interval': 200,
            'margin': 0.6,
            'list_only': list_only
        },
        lr={'contrastive': 1e-4},
    )

def scopev2_experiments(dev, list_only=False):
    torch.cuda.set_device(dev)
    run_experiments(
        common_kwargs = {
            'dataset': 'scope_v2',
            'save': True,
            'k': 100,
            'stride': 80,
            'n_epochs': 201,
            'interval': 200,
            'margin': 0.6,
            'list_only': list_only,
            'cls': 'all',
            'batch_size': 256,
        },
        lr={'contrastive': 1e-4},
    )

def run_experiments(common_kwargs, lr):
    pooling_method = [
        'bom', 
        'cls', 
        'avg',
        'sep'
    ]
    emb_models = [
        'prottrans', 
        'protbert',
        'esm2-35M', 
        'esm2-150M',
        'esm2-650M'
    ]
    seeds = [
        261,
        2602,
        26003,
        2604,
        265
    ]
    for seed in seeds:
        common_kwargs['seed'] = seed
        for pm in pooling_method:
            for em in emb_models:           
                if 'regression' in lr.keys():
                    run_regression(**common_kwargs, pooling_mode=pm, emb_model=em, lr=lr['regression'])  
                if 'preference' in lr.keys():
                    run_preference(**common_kwargs, pooling_mode=pm, emb_model=em, lr=lr['preference'])
                if 'contrastive' in lr.keys():
                    run_contrastive(**common_kwargs, pooling_mode=pm, emb_model=em, lr=lr['contrastive'])



if __name__ == '__main__':
    list_only = False
    stability_experiments(dev=0, list_only=list_only)
    solubility_experiments(dev=0, list_only=list_only)
    beta_lactamase_experiments(dev=0, list_only=list_only)
    fluorescence_experiments(dev=0, list_only=list_only)
    ppi_experiments(dev=0, list_only=list_only)
    dpi_experiments(dev=1, list_only=list_only)
    scope_experiments(dev=0, list_only=list_only) 
    scopev2_experiments(dev=1, list_only=list_only) 