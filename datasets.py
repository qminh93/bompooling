from utils import *
from tape.datasets import LMDBDataset
from torchdrug import datasets
from torchdrug import transforms
from plm_wrapper import *

DATA_ROOT = ""
DATA_DIR = {
    'peer_root': f'{DATA_ROOT}/protein-datasets',
    'stability': f'{DATA_ROOT}/stability',
    'fluorescence': f'{DATA_ROOT}/fluorescence',
    'beta_lactamase': f'{DATA_ROOT}/protein-datasets/beta_lactamase',
    'solubility': f'{DATA_ROOT}/protein-datasets/solubility',
    'dpi': f'{DATA_ROOT}/protein-peptide/data',
    'ppi': f'{DATA_ROOT}/protein-protein',
    'scope': f'{DATA_ROOT}/SCOPe',
    'scope_v2': f'{DATA_ROOT}/SCOPe_v2'
}
DPI_DOMAINS = [
    'PTB', 
    'PTP', 
    'Kinase_TK', 
]
TEST_DPI_DOMAINS = {
    'PTB': 13619,
    'PTP': 69252,
    'Kinase_TK': 136229
}

class RegressionDataset:
    def __init__(self, **kwargs):
        self.save = kwargs.get('save', 'load_pool')
        self.batch_size = kwargs.get('embedding_batch_size', 512)
        self.pooling_mode = kwargs.get('pooling_mode', 'avg')
        raw_data_dir, pool_data_dir = self.data_dir(**kwargs)
        print(f'Loading {self.dataset} dataset')
        if self.save == 'save':
            self.emb_model = PLM[kwargs.get('emb_model', 'prottrans')]()
            self.embed_data(**kwargs)
            torch.save(self.make_data_dict(), raw_data_dir)
            self.filter_data(**kwargs)
            self.pool_data(**kwargs)
            torch.save(self.make_data_dict(), pool_data_dir)
        elif self.save == 'load_raw':
            self.load_data_dict(torch.load(raw_data_dir))
            self.filter_data(**kwargs)
            self.pool_data(**kwargs)
            torch.save(self.make_data_dict(), pool_data_dir)
        elif self.save == 'load_pool':
            self.load_data_dict(torch.load(pool_data_dir))
        
        self.n_train = len(self.x_train)
        self.n_test = len(self.x_test)
        self.y_train = (self.y_train - self.y_train.min()) / (self.y_train.max() - self.y_train.min())
        self.y_test = (self.y_test - self.y_train.min()) / (self.y_train.max() - self.y_train.min())

    def data_dir(self, **kwargs):
        raise NotImplementedError

    def embed_data(self, **kwargs):
        raise NotImplementedError
    
    def filter_data(self, **kwargs):
        min_len = kwargs.get('min_len', None)
        if min_len is None:
            return
        
        filter_train_idx, filter_test_idx = [], []
        for i, x in enumerate(self.x_train):
            if x.shape[0] >= min_len:
                filter_train_idx.append(i)
        
        for i, x in enumerate(self.x_test):
            if x.shape[0] >= min_len:
                filter_test_idx.append(i)

        self.x_train = [self.x_train[i] for i in filter_train_idx]
        self.y_train = self.y_train[torch.tensor(filter_train_idx)]
        self.x_test = [self.x_test[i] for i in filter_test_idx]
        self.y_test = self.y_test[torch.tensor(filter_test_idx)]

    def make_data_dict(self):
        return {
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'y_test': self.y_test
        }

    def load_data_dict(self, data_dict):
        self.x_train = data_dict['x_train']
        self.y_train = data_dict['y_train']
        self.x_test = data_dict['x_test']
        self.y_test = data_dict['y_test']

    def pool_data(self, **kwargs):
        pooling_mode = kwargs['pooling_mode']
        if pooling_mode == 'cls':
            self.cls_pooling()
        if pooling_mode == 'avg':
            self.avg_pooling()
        if pooling_mode == 'bom':
            self.bom_pooling(kwargs['k'], kwargs['stride'])

    def cls_pooling(self):
        print(f'Performing CLS Pooling')
        self.x_train = [x[0, :] for x in tqdm(self.x_train)]
        self.x_test = [x[0, :] for x in tqdm(self.x_test)]
        gc.collect()

    def avg_pooling(self):
        print(f'Performing Avg Pooling')
        self.x_train = [x[1:-1, :].mean(dim=0) for x in tqdm(self.x_train)]
        self.x_test = [x[1:-1, :].mean(dim=0) for x in tqdm(self.x_test)]
        gc.collect()

    def bom_pooling(self, k, stride):
        print(f'Performing BoM Pooling')
        self.x_train = [x[1:-1, :].unfold(0, k, stride).mean(dim=-1) for x in tqdm(self.x_train)]
        self.x_test = [x[1:-1, :].unfold(0, k, stride).mean(dim=-1) for x in tqdm(self.x_test)]
        gc.collect()

    def generate_ranking_test_set(self, save=None):
        left, right = torch.randperm(self.n_test), torch.randperm(self.n_test)
        label = (self.y_test[left] > self.y_test[right]).int()
        test_set = {'left': left, 'right': right, 'label': label}
        if save is not None:
            torch.save(test_set, save)
        return test_set

class TapeRegressionDataset(RegressionDataset):
    def __init__(self, **kwargs):
        self.dataset = kwargs.get('dataset', 'fluorescence')
        assert self.dataset == 'fluorescence' or self.dataset == 'stability', 'Not a TAPE dataset'
        super(TapeRegressionDataset, self).__init__(**kwargs)

    def data_dir(self, **kwargs):
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        emb_model = kwargs.get('emb_model', 'prottrans')
        raw_data_dir = f'{DATA_DIR[self.dataset]}/{self.dataset}_{emb_model}.pt'
        pool_data_dir = f'{DATA_DIR[self.dataset]}/{self.dataset}_{emb_model}_{self.pooling_mode}{suffix}.pt'
        return raw_data_dir, pool_data_dir
        
            
    def embed_data(self, **kwargs):
        input_var = 'primary'
        target_var = 'log_fluorescence' if self.dataset == 'fluorescence' else 'stability_score'
        self.x_train, self.y_train = self.embed_lmdb(
            f'{DATA_DIR[self.dataset]}/{self.dataset}_train.lmdb',
            input_var, target_var
        )
        self.x_test, self.y_test = self.embed_lmdb(
            f'{DATA_DIR[self.dataset]}/{self.dataset}_test.lmdb',
            input_var, target_var
        )


    def embed_lmdb(self, lmdb_file, input_var, target_var):
        raw = LMDBDataset(lmdb_file)
        n_data = len(raw)
        x, y = [], []
        with torch.no_grad():
            bar = trange(0, n_data, self.batch_size)
            bar.set_description_str(f'Load and embed data from {lmdb_file}')
            for bid in bar:
                batch, seq_len = [], []
                for i in range(bid, min(bid + self.batch_size, n_data)):
                    batch.append(raw[i][input_var])
                    seq_len.append(len(raw[i][input_var]))
                    y.append(raw[i][target_var][0])
                batch_emb = self.emb_model(batch)
                for i in range(batch_emb.shape[0]):
                    x.append(batch_emb[i][:seq_len[i] + 2].to('cpu'))
            y = torch.tensor(y)
        return x, y
    
class PEERRegressionDataset(RegressionDataset):
    def __init__(self, **kwargs):
        self.dataset = kwargs.get('dataset', 'beta_lactamase')
        assert self.dataset == 'beta_lactamase' or self.dataset == 'solubility', 'Not a PEER dataset'
        super(PEERRegressionDataset, self).__init__(**kwargs)

    def data_dir(self, **kwargs):
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        emb_model = kwargs.get('emb_model', 'protbert')
        raw_data_dir = f'{DATA_DIR[self.dataset]}/{self.dataset}_{emb_model}.pt'
        pool_data_dir = f'{DATA_DIR[self.dataset]}/{self.dataset}_{emb_model}_{self.pooling_mode}{suffix}.pt'
        return raw_data_dir, pool_data_dir
        

    def embed_data(self, **kwargs):
        truncate_transform = transforms.TruncateProtein(max_length=200, random=False)
        protein_view_transform = transforms.ProteinView(view="residue")
        transform = transforms.Compose([truncate_transform, protein_view_transform])
        if self.dataset == 'beta_lactamase':
            raw = datasets.BetaLactamase(
                DATA_DIR['peer_root'], atom_feature=None, bond_feature=None, 
                residue_feature='default', transform=transform
            )
        if self.dataset == 'solubility':
            raw = datasets.Solubility(
                DATA_DIR['peer_root'], atom_feature=None, bond_feature=None, 
                residue_feature='default', transform=transform
            )
        n_data = len(raw)
        n_test = int(kwargs.get('test_ratio', 0.1) * n_data)
        x, y = [], []
        with torch.no_grad():
            bar = trange(0, len(raw), self.batch_size)
            bar.set_description_str(f'Load and embed {self.dataset} data')
            for bid in bar:
                batch, seq_len = [], []
                for i in range(bid, min(bid + self.batch_size, n_data)):
                    batch.append(raw[i]['graph'].to_sequence().replace('.', ' '))
                    seq_len.append((len(batch[-1]) + 1) // 2)
                    y.append(raw[i][raw.target_fields[0]])
                batch_emb = self.emb_model(batch)
                for i in range(batch_emb.shape[0]):
                    x.append(batch_emb[i][:seq_len[i] + 2].to('cpu'))
        y = torch.tensor(y)
        idx = torch.randperm(n_data)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        self.x_test, self.x_train = [x[i] for i in test_idx], [x[i] for i in train_idx]
        self.y_test, self.y_train = y[test_idx], y[train_idx]

class PairedDataset:
    def __init__(self, **kwargs):
        self.emb_model = kwargs.get('emb_model', 'protbert')
        self.pooling_mode = kwargs.get('pooling_mode', 'avg')
        self.save = kwargs.get('save', 'load_pool')
        data_dir = self.data_dir(**kwargs)
        if self.save == 'save':
            self.emb_net = PLM[self.emb_model]()
            self.make_train(**kwargs)
            self.make_test(**kwargs)
            torch.save(self.train_data, data_dir['raw_train'])
            torch.save(self.test_data, data_dir['raw_test'])
            self.pool_data(**kwargs)
            torch.save(self.train_data, data_dir['pool_train'])
            torch.save(self.test_data, data_dir['pool_test'])
        elif self.save == 'load_raw':
            self.train_data = torch.load(data_dir['raw_train'], map_location='cpu')
            self.test_data = torch.load(data_dir['raw_test'], map_location='cpu')
            self.pool_data(**kwargs)
            torch.save(self.train_data, data_dir['pool_train'])
            torch.save(self.test_data, data_dir['pool_test'])
        elif self.save == 'load_raw_only':
            self.train_data = torch.load(data_dir['raw_train'], map_location='cpu')
            self.test_data = torch.load(data_dir['raw_test'], map_location='cpu')
        elif self.save == 'load_pool':
            self.train_data = torch.load(data_dir['pool_train'], map_location='cpu')
            self.test_data = torch.load(data_dir['pool_test'], map_location='cpu')
    
    def data_dir(self, **kwargs):
        raise NotImplementedError
    
    def make_train(self, **kwargs):
        raise NotImplementedError
    
    def make_test(self, **kwargs):
        raise NotImplementedError

    def pool_data(self, **kwargs):
        raise NotImplementedError
    
    def create_triplet_batch(self, anchor):
        raise NotImplementedError

class DPIDataset(PairedDataset):
    def __init__(self, **kwargs):
        super(DPIDataset, self).__init__(**kwargs)
    
    def data_dir(self, **kwargs):
        data_dir = {}
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        data_dir['raw_train'] = f'{DATA_DIR["dpi"]}/train_{self.emb_model}.pt'
        data_dir['pool_train'] = f'{DATA_DIR["dpi"]}/train_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        data_dir['raw_test'] = f'{DATA_DIR["dpi"]}/test_{self.emb_model}.pt'
        data_dir['pool_test'] = f'{DATA_DIR["dpi"]}/test_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        return data_dir

    def make_train(self, **kwargs):
        print(f'Preprocessing training data')
        train_data = {}
        train_data['doms'], train_data['peps'], train_data['pos'], train_data['neg'] = self.load_paired_data('train', minlen=kwargs['minlen'])
        with torch.no_grad():
            train_data['dom_emb'] = [self.emb_net([d]) for d in tqdm(train_data['doms'])]
            train_data['pep_emb'] = [self.emb_net([p]) for p in tqdm(train_data['peps'])]
        train_data['anchors'] = np.array([
            i for i in range(len(train_data['doms']))
            if len(train_data['pos'][i]) and len(train_data['neg'][i])
        ])
        self.train_data = train_data
        

    def make_test(self, **kwargs):
        test_data = {}
        for test_fold in DPI_DOMAINS:
            print(f'Preprocessing test data in {test_fold} domain')
            fold_data = {}
            fold_data['doms'], fold_data['peps'], fold_data['pos'], fold_data['neg'] = self.load_paired_data(test_fold, minlen=kwargs['minlen'])
            with torch.no_grad():
                fold_data['dom_emb'] = [self.emb_net([d]) for d in tqdm(fold_data['doms'])]
                fold_data['pep_emb'] = [self.emb_net([p]) for p in tqdm(fold_data['peps'])]
            test_data[test_fold] = fold_data
        self.test_data = test_data

    def load_paired_data(self, fold, minlen=0):
        if fold == 'train':
            f = open(f'{DATA_DIR["dpi"]}/data_without_processed_duplicates/preprocessed_raw_data.csv')
        else:
            f = open(f'{DATA_DIR["dpi"]}/data_without_processed_duplicates/raw_data/{fold}.csv')
        lines = f.readlines()[0 if fold == 'train' else 1:]
        doms, peps = set(), set()
        for line in lines:
            tokens = line.split(',')
            dom, pep = tokens[1], tokens[3 if fold == 'train' else 2]
            dom, pep = re.sub(r"[-y]", "", dom), re.sub(r"[-y]", "", pep)
            if len(dom) > minlen:
                doms.add(dom)
                peps.add(pep)
                
        doms, peps = list(doms), list(peps)
        dom_id = {d: i for i, d in enumerate(doms)}
        pep_id = {p: i for i, p in enumerate(peps)}
        pos = [set() for _ in range(len(doms))]
        neg = [set() for _ in range(len(doms))]
        for line in lines:
            tokens = line.split(',')
            dom, pep, label = tokens[1], tokens[3 if fold == 'train' else 2], int(tokens[-1])
            dom, pep = re.sub(r"[-y]", "", dom), re.sub(r"[-y]", "", pep)
            if len(dom) > minlen:
                (pos if label else neg)[dom_id[dom]].add(pep_id[pep])
        return doms, peps, [list(p) for p in pos], [list(n) for n in neg]

    def pool_data(self, **kwargs):
        print(f'Performing Avg Pooling for short peptide chains')
        self.train_data['pep_emb'] = [x[0, 1:-1, :].mean(dim=0, keepdim=(self.pooling_mode=='bom')) for x in tqdm(self.train_data['pep_emb'])]
        for test_fold in DPI_DOMAINS:
            self.test_data[test_fold]['pep_emb'] = [x[0, 1:-1, :].mean(dim=0, keepdim=(self.pooling_mode=='bom')) for x in tqdm(self.test_data[test_fold]['pep_emb'])]

        if kwargs['pooling_mode'] == 'cls':
            print(f'Performing CLS Pooling for domains')
            self.train_data['dom_emb'] = [x[0, 0, :] for x in tqdm(self.train_data['dom_emb'])]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [x[0, 0, :] for x in tqdm(self.test_data[test_fold]['dom_emb'])]
                gc.collect()

        if kwargs['pooling_mode'] == 'sep':
            print(f'Performing SEP Pooling for domains')
            self.train_data['dom_emb'] = [x[0, -1, :] for x in tqdm(self.train_data['dom_emb'])]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [x[0, -1, :] for x in tqdm(self.test_data[test_fold]['dom_emb'])]
                gc.collect()

        if kwargs['pooling_mode'] == 'avg':
            print(f'Performing Avg Pooling for domains')
            self.train_data['dom_emb'] = [x[0, 1:-1, :].mean(dim=0) for x in tqdm(self.train_data['dom_emb'])]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [x[0, 1:-1, :].mean(dim=0) for x in tqdm(self.test_data[test_fold]['dom_emb'])]
                gc.collect()

        if kwargs['pooling_mode'] == 'bom':
            print(f'Performing BoM Pooling for domains')
            self.train_data['dom_emb'] = [
                x[0, 1:-1, :].unfold(0, min(kwargs['k'], x.shape[1]-2), kwargs['stride']).mean(dim=-1) 
                for x in tqdm(self.train_data['dom_emb'])
            ]
            for test_fold in DPI_DOMAINS:
                self.test_data[test_fold]['dom_emb'] = [
                    x[0, 1:-1, :].unfold(0, min(kwargs['k'], x.shape[1]-2), kwargs['stride']).mean(dim=-1) 
                    for x in tqdm(self.test_data[test_fold]['dom_emb'])
                ]
                gc.collect()
        
    def create_triplet_batch(self, anchor):
        pos_emb = [self.train_data['pep_emb'][np.random.choice(self.train_data['pos'][a])].to('cuda') for a in anchor]
        neg_emb = [self.train_data['pep_emb'][np.random.choice(self.train_data['neg'][a])].to('cuda') for a in anchor]
        anchor_emb = [self.train_data['dom_emb'][a].to('cuda') for a in anchor]
        return anchor_emb, neg_emb, pos_emb

class SCOPeDataset(PairedDataset):
    def __init__(self, **kwargs):
        super(SCOPeDataset, self).__init__(**kwargs)

    def data_dir(self, **kwargs):
        data_dir = {}
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        data_dir['raw_train'] = f'{DATA_DIR["scope"]}/train_{self.emb_model}.pt'
        data_dir['pool_train'] = f'{DATA_DIR["scope"]}/train_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        data_dir['raw_test'] = f'{DATA_DIR["scope"]}/test_{self.emb_model}.pt'
        data_dir['pool_test'] = f'{DATA_DIR["scope"]}/test_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        return data_dir
    
    def make_train(self, **kwargs):
        train_data = {}
        train_data['meta_info'], seqs = self.generate_metainfo(**kwargs)
        train_data['seq_emb'] = []
        with torch.no_grad():
            n_seq = len(seqs)
            bar = trange(0, n_seq, 32)
            for bid in bar:
                seq_len, batch = [], []
                for i in range(bid, min(bid + 32, n_seq)):
                    seq_len.append(len(seqs[i]))
                    batch.append(seqs[i])
                batch_emb = self.emb_net(batch)
                for i in range(batch_emb.shape[0]):
                    train_data['seq_emb'].append(batch_emb[i][:seq_len[i] + 2].to('cpu'))
                gc.collect()
        
        all_folds = list(train_data['meta_info']['folds'].keys())
        np.random.shuffle(all_folds)
        num_test_fold = int(len(all_folds) * kwargs.get('test_ratio', 0.05))
        train_fold, test_fold = all_folds[num_test_fold:], all_folds[:num_test_fold]
        train_data['anchors'], test_data = [], []
        for sid, (fold, sf, _, _) in train_data['meta_info']['membership'].items():
            if (train_data['meta_info']['superfam_size'][sf] >= 2) and (fold in train_fold):
                train_data['anchors'].append(sid)
            elif fold in test_fold:
                test_data.append(sid)
        train_data['anchors'] = np.array(train_data['anchors'])
        self.test_data = np.array(test_data)
        self.train_data = train_data
    

    def make_test(self, **kwargs):
        # DO NOTHING, ALREADY GENERATED IN MAKE_TRAIN
        pass

    def generate_metainfo(self, **kwargs):
        min_len = 150
        data_file = open(f'{DATA_DIR["scope"]}/SCOPe_metadata.csv')
        data = data_file.readlines()[1:]
        meta_info = {
            'folds': defaultdict(set),
            'superfams': defaultdict(set),
            'fams': defaultdict(set),
            'membership': {},
            'fold_size': defaultdict(lambda: 0),
            'superfam_size': defaultdict(lambda: 0),
            'fam_size': defaultdict(lambda: 0),
        }
        seqs = []
        n_seq = 0
        for d in data:
            tokens = d.split(',')
            try:
                sid, _, seq, seq_len = tokens[0], tokens[1], tokens[2], int(tokens[3])
                fold, superfam, fam = tokens[4], tokens[5], tokens[6]
                if seq_len >= min_len:
                    meta_info['folds'][fold].add(superfam)
                    meta_info['superfams'][superfam].add(fam)
                    meta_info['fams'][fam].add(sid)
                    meta_info['membership'][sid] = (fold, superfam, fam, n_seq)
                    meta_info['fold_size'][fold] += 1
                    meta_info['superfam_size'][superfam] += 1
                    meta_info['fam_size'][fam] += 1
                    seqs.append(seq.strip())
                    n_seq += 1
            except Exception as e:
                continue
        for k, v in meta_info.items():
            meta_info[k] = dict(v)
        return meta_info, seqs

    def pool_data(self, **kwargs):
        if kwargs['pooling_mode'] == 'cls':
            print(f'Performing CLS Pooling')
            self.train_data['seq_emb'] = [x[0, :] for x in tqdm(self.train_data['seq_emb'])]
        if kwargs['pooling_mode'] == 'avg':
            print(f'Performing Avg Pooling')
            self.train_data['seq_emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.train_data['seq_emb'])]
        if kwargs['pooling_mode'] == 'bom':
            print(f'Performing BoM Pooling')
            self.train_data['seq_emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.train_data['seq_emb'])]
        gc.collect()

    def create_triplet_batch(self, anchor):
        anchor_emb, neg_emb, pos_emb = [], [], []
        for a in anchor:
            _, sf, _, anchor_id = self.train_data['meta_info']['membership'][a]
            # Positive sampling
            while True:
                pos_fam = np.random.choice(list(self.train_data['meta_info']['superfams'][sf]))
                pos_seq = np.random.choice(list(self.train_data['meta_info']['fams'][pos_fam]))
                if pos_seq != a:
                    _, _, _, pos_id = self.train_data['meta_info']['membership'][pos_seq]
                    pos_emb.append(self.train_data['seq_emb'][pos_id].to('cuda'))
                    break
            # Negative sampling
            while True:
                neg_seq = np.random.choice(self.train_data['anchors'])
                _, neg_sf, _, _ = self.train_data['meta_info']['membership'][neg_seq]
                if neg_sf != sf:
                    _, _, _, neg_id = self.train_data['meta_info']['membership'][neg_seq]
                    neg_emb.append(self.train_data['seq_emb'][neg_id].to('cuda'))
                    break
            anchor_emb.append(self.train_data['seq_emb'][anchor_id].to('cuda'))
        return anchor_emb, neg_emb, pos_emb


class SCOPev2Dataset(PairedDataset):
    def __init__(self, **kwargs):
        super(SCOPev2Dataset, self).__init__(**kwargs)
        self.train_adj_list = torch.load(f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/train_scope_{kwargs["cls"]}_adjlist.pt')
        self.train_data['anchors'] = np.array([
                i for i in range(len(self.train_adj_list)) 
                if (len(self.train_adj_list[i]) > 1)
                and (len(self.train_adj_list[i]) < 200)
            ]
        )
        self.test_adj_list = torch.load(f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/test_scope_{kwargs["cls"]}_adjlist.pt')


    def data_dir(self, **kwargs):
        data_dir = {}
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        data_dir['raw_train'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/train_{self.emb_model}.pt'
        data_dir['pool_train'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/train_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        data_dir['raw_test'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/test_{self.emb_model}.pt'
        data_dir['pool_test'] = f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/test_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        return data_dir
    
    def make_fold(self, fold, emb_batch_size, **kwargs):
        raw = torch.load(f'{DATA_DIR["scope_v2"]}/class_{kwargs["cls"]}/{fold}_scope_data_{kwargs["cls"]}.pt')
        emb = []
        n = len(raw['seq_list'])
        with torch.no_grad():
            for i in trange(0, n, emb_batch_size):
                batch_id = raw['seq_list'][i: min(i + emb_batch_size, n)]
                batch = [raw['seq_dict'][sf][fa][j] for sf, fa, j in batch_id]
                batch_emb = self.emb_net(batch).to('cpu')
                emb += [batch_emb[j, :len(batch[j]) + 2, :] for j in range(batch_emb.shape[0])]
                gc.collect()
            
        return emb, raw['seq_list']

    def make_train(self, **kwargs):
        emb, seq_list = self.make_fold('train', 64, **kwargs)
        self.train_data = {'emb': emb, 'seq_list': seq_list}
    
    def make_test(self, **kwargs):
        emb, seq_list = self.make_fold('test', 64, **kwargs)
        self.test_data = {'emb': emb, 'seq_list': seq_list}

    
    def pool_data(self, **kwargs):
        if kwargs['pooling_mode'] == 'cls':
            print(f'Performing CLS Pooling')
            self.train_data['emb'] = [x[0, :] for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[0, :] for x in tqdm(self.test_data['emb'])]

        if kwargs['pooling_mode'] == 'avg':
            print(f'Performing Avg Pooling')
            self.train_data['emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.test_data['emb'])]

        if kwargs['pooling_mode'] == 'sep':
            print(f'Performing SEP Pooling')
            self.train_data['emb'] = [x[-1, :] for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[-1, :] for x in tqdm(self.test_data['emb'])]

        if kwargs['pooling_mode'] == 'bom':
            print(f'Performing BoM Pooling')
            self.train_data['emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.train_data['emb'])]
            self.test_data['emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.test_data['emb'])]

        gc.collect()

    def create_triplet_batch(self, anchor):
        anchor_emb, neg_emb, pos_emb = [], [], []
        for idx in anchor:
            anchor_emb.append(self.train_data['emb'][idx].to('cuda'))
            
            # Positive sampling
            pos_idx = np.random.choice(self.train_adj_list[idx])
            pos_emb.append(self.train_data['emb'][pos_idx].to('cuda'))
            
            # Negative sampling
            neg_idx = pos_idx
            while neg_idx in self.train_adj_list[idx]:
                neg_idx = np.random.choice(len(self.train_adj_list))
            neg_emb.append(self.train_data['emb'][neg_idx].to('cuda'))
                
        return anchor_emb, neg_emb, pos_emb

class PPIDataset(PairedDataset):
    def __init__(self, **kwargs):
        super(PPIDataset, self).__init__(**kwargs)
        self.train_data['anchors'] = np.array([
            i for i in range(len(self.train_data['pos']))
            if len(self.train_data['pos'][i]) and len(self.train_data['neg'][i])
        ])

    def data_dir(self, **kwargs):
        data_dir = {}
        suffix = f'_k{kwargs["k"]}_s{kwargs["stride"]}' if self.pooling_mode == 'bom' else ''
        data_dir['raw_train'] = f'{DATA_DIR["ppi"]}/train_{self.emb_model}.pt'
        data_dir['pool_train'] = f'{DATA_DIR["ppi"]}/train_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        data_dir['raw_test'] = f'{DATA_DIR["ppi"]}/test_{self.emb_model}.pt'
        data_dir['pool_test'] = f'{DATA_DIR["ppi"]}/test_{self.emb_model}_{self.pooling_mode}{suffix}.pt'
        return data_dir

    def make_train(self, **kwargs):
        print(f'Embedding PPI training data')
        pos_file, neg_file, seq_file = f'{DATA_DIR["ppi"]}/Intra0_pos_rr.txt', f'{DATA_DIR["ppi"]}/Intra0_neg_rr.txt', f'{DATA_DIR["ppi"]}/train_seqs.pt'
        self.train_data = self.load_paired_data(pos_file, neg_file, seq_file)

    def make_test(self, **kwargs):
        print(f'Embedding PPI test data')
        pos_file, neg_file, seq_file = f'{DATA_DIR["ppi"]}/Intra2_pos_rr.txt', f'{DATA_DIR["ppi"]}/Intra2_neg_rr.txt', f'{DATA_DIR["ppi"]}/test_seqs.pt'
        self.test_data = self.load_paired_data(pos_file, neg_file, seq_file)

    def load_paired_data(self, pos_file, neg_file, seq_file):
        f_pos, f_neg = open(pos_file), open(neg_file)
        data = {}
        seqs = torch.load(seq_file)
        seqs = {uid: seq for uid, seq in seqs.items() if (len(seq) > 150) and (len(seq) < 1024)}

        uid_to_idx = {uid: i for i, uid in enumerate(seqs.keys())}
        data['pos'] = [[] for _ in range(len(seqs))]
        data['neg'] = [[] for _ in range(len(seqs))]
        data['seq_emb'] = [[] for _ in range(len(seqs))]

        for line in f_pos.readlines():
            s1, s2 = line.strip().split(' ')
            if (s1 not in seqs) or (s2 not in seqs):
                continue
            else:
                i1, i2 = uid_to_idx[s1], uid_to_idx[s2]
                data['pos'][i1].append(i2)
                data['pos'][i2].append(i1)
        
        for line in f_neg.readlines():
            s1, s2 = line.strip().split(' ')
            if (s1 not in seqs) or (s2 not in seqs):
                continue
            else:
                i1, i2 = uid_to_idx[s1], uid_to_idx[s2]
                data['neg'][i1].append(i2)
                data['neg'][i2].append(i1)
        
        seqs = list(seqs.values())
        batch_size = 32
        with torch.no_grad():
            n_seq = len(seqs)
            bar = trange(0, n_seq, batch_size)
            data['seq_emb'] = []
            for bid in bar:
                seq_len, batch = [], []
                for i in range(bid, min(bid + batch_size, n_seq)):
                    seq_len.append(len(seqs[i]))
                    batch.append(seqs[i])
                batch_emb = self.emb_net(batch).to('cpu')
                for i in range(batch_emb.shape[0]):
                    data['seq_emb'].append(batch_emb[i][:seq_len[i] + 2])
                gc.collect()
        return data

    def pool_data(self, **kwargs):
        if kwargs['pooling_mode'] == 'cls':
            print(f'Performing CLS Pooling')
            self.train_data['seq_emb'] = [x[0, :] for x in tqdm(self.train_data['seq_emb'])]
            self.test_data['seq_emb'] = [x[0, :] for x in tqdm(self.test_data['seq_emb'])]
        if kwargs['pooling_mode'] == 'avg':
            print(f'Performing Avg Pooling')
            self.train_data['seq_emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.train_data['seq_emb'])]
            self.test_data['seq_emb'] = [x[1:-1, :].mean(dim=0) for x in tqdm(self.test_data['seq_emb'])]
        if kwargs['pooling_mode'] == 'bom':
            print(f'Performing BoM Pooling')
            self.train_data['seq_emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.train_data['seq_emb'])]
            self.test_data['seq_emb'] = [x[1:-1, :].unfold(0, kwargs['k'], kwargs['stride']).mean(dim=-1) for x in tqdm(self.test_data['seq_emb'])]
        gc.collect()
    
    def create_triplet_batch(self, anchor):
        pos_emb = [self.train_data['seq_emb'][np.random.choice(self.train_data['pos'][a])].to('cuda') for a in anchor]
        # neg_emb = [self.train_data['seq_emb'][np.random.choice(self.train_data['neg'][a])].to('cuda') for a in anchor]
        neg_emb = [self.train_data['seq_emb'][np.random.choice(len(self.train_data['seq_emb']))].to('cuda') for a in anchor]
        anchor_emb = [self.train_data['seq_emb'][a].to('cuda') for a in anchor]
        return anchor_emb, neg_emb, pos_emb


def generate_training_data(task, k, stride):
    pooling_mode = [
        'bom', 
        'sep',
        'cls', 
        'avg'
    ]
    embedding_model = [
        'protbert', 
        'prottrans', 
        'esm2-35M', 
        'esm2-150M',
        'esm2-650M'
    ]
    for em in embedding_model:
        for pm in pooling_mode:
            data = None
            if task in ['fluorescence', 'stability']:
                data = TapeRegressionDataset(
                    dataset=task,
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride
                )
                for i in range(5):
                    data.generate_ranking_test_set(f'{DATA_DIR[task]}/{task}_preference_test{i}.pt')
            elif task in ['beta_lactamase', 'solubility']:
                data = PEERRegressionDataset(
                    dataset=task,
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride,
                    min_len=50
                )
                for i in range(5):
                    data.generate_ranking_test_set(f'{DATA_DIR[task]}/{task}_preference_test{i}.pt')
            elif task == 'scope':
                data = SCOPeDataset(
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride,
                )
            elif task == 'scope_v2':
                data = SCOPev2Dataset(
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride,
                    cls='all'
                )
            elif task == 'dpi':
                data = DPIDataset(
                    emb_model=em,
                    save='save' if pm == '!bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride,
                    minlen=0,
                )
            elif task == 'ppi':
                data = PPIDataset(
                    emb_model=em,
                    save='save' if pm == 'bom' else 'load_raw',
                    pooling_mode=pm,
                    k=k, stride=stride
                )    

if __name__ == '__main__':
    generate_training_data('fluorescence', 100, 20)
    generate_training_data('beta_lactamase', 100, 20)
    generate_training_data('dpi', 40, 8)
    generate_training_data('scope_v2', 100, 80)