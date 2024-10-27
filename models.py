from datasets import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

class RegressionModel:
    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.pooling_net = kwargs['pool_net']
        self.predictor_net = kwargs['pred_net']
        self.batch_size = kwargs['batch_size']
        assert isinstance(self.data, RegressionDataset), 'Not a regression dataset'
        self.history = None

    def reset_history(self):
        self.history = {
            'epochs': [],
            'error': [],
            'pred': []
        }

    def predict(self, x_batch):
        return self.predictor_net(self.pooling_net(x_batch))

    def loss(self, x_batch, y_batch, reduction='sum'):
        y_pred = self.predict(x_batch).flatten()
        return y_pred, F.mse_loss(y_pred, y_batch, reduction=reduction)

    def evaluate(self):
        loss = 0.
        with torch.no_grad():
            self.pooling_net.eval()
            self.predictor_net.eval()
            pred = []
            for i in range(0, self.data.n_test, self.batch_size):
                x_batch = [x.to('cuda') for x in self.data.x_test[i: min(i+self.batch_size, self.data.n_test)]]
                y_batch = self.data.y_test[i: min(i+self.batch_size, self.data.n_test)].to('cuda')
                y_pred, batch_loss = self.loss(x_batch, y_batch)
                loss += batch_loss
                pred.append(y_pred)
        return pred, loss / self.data.n_test

    def train(self, n_epochs=10001, interval=100, lr=1e-4):
        opt = AdamW(list(self.pooling_net.parameters()) + list(self.predictor_net.parameters()), lr=lr)
        for ep in range(n_epochs):
            if ep % interval == 0:
                pred, loss = self.evaluate()
                self.history['epochs'].append(ep)
                self.history['error'].append(loss.item())
                self.history['pred'].append(pred)
                print(f'Epoch {ep} Test Loss={loss.item():.5f}\n')
            self.pooling_net.train()
            self.predictor_net.train()
            idx = torch.randperm(self.data.n_train)
            bar = trange(0, self.data.n_train, self.batch_size)
            bar.set_description_str(f'Epoch {ep}:')
            for i in bar:
                opt.zero_grad()
                idx_batch = idx[i: min(i + self.batch_size, self.data.n_train)]
                x_batch = [self.data.x_train[j].to('cuda') for j in idx_batch]
                y_batch = self.data.y_train[idx_batch].to('cuda')
                _, loss = self.loss(x_batch, y_batch, reduction='mean')
                bar.set_postfix_str(f'Batch Loss = {loss.item():.5f}')
                loss.backward()
                opt.step()


class PreferenceModel:
    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.pooling_net = kwargs['pool_net']
        self.rank_net = kwargs['rank_net']
        self.batch_size = kwargs['batch_size']
        self.test_sets = kwargs['test_sets']
        assert isinstance(self.data, RegressionDataset), 'Not a regression dataset'
        self.history = None

    def reset_history(self):
        self.history = {
            'epochs': [],
            'error': [],
            'prob': []
        }

    def rank(self, l, r):
        return self.rank_net(self.pooling_net(l, r))

    def loss(self, l, r, y, reduction='sum'):
        diff = self.rank(l, r).flatten()
        return F.binary_cross_entropy_with_logits(diff, y, reduction=reduction)

    def evaluate(self, test_set_path):
        test_set = torch.load(test_set_path)
        loss = 0.
        prob = []
        with torch.no_grad():
            self.pooling_net.eval()
            self.rank_net.eval()    
            for i in range(0, self.data.n_test, self.batch_size):
                l_idx = test_set['left'][i: min(i+self.batch_size, self.data.n_test)]
                r_idx = test_set['right'][i: min(i + self.batch_size, self.data.n_test)]
                l_batch = [self.data.x_test[j].to('cuda') for j in l_idx]
                r_batch = [self.data.x_test[j].to('cuda') for j in r_idx]
                prob.append(torch.sigmoid(self.rank(l_batch, r_batch)).flatten())
                pred = (prob[-1] > 0.5).int()
                y_batch = (self.data.y_test[l_idx] > self.data.y_test[r_idx]).int().to('cuda')
                loss += (pred != y_batch).sum()
        loss = loss / self.data.n_test
        return prob, loss

    def train(self, n_epochs=10001, interval=100, lr=1e-3):
        opt = AdamW(list(self.pooling_net.parameters()) + list(self.rank_net.parameters()), lr=lr)
        for ep in range(n_epochs):
            if ep % interval == 0:
                self.history['epochs'].append(ep)
                self.history['error'].append([])
                self.history['prob'].append([])
                for test_set in self.test_sets:
                    prob, loss = self.evaluate(test_set)
                    self.history['error'][-1].append(loss.item())
                    self.history['prob'][-1].append(prob)
                print(f'Epoch {ep} '
                      f'Avg Test Error={np.mean(self.history["error"][-1]):.5f} '
                      f'Std={np.std(self.history["error"][-1]):.5f}\n')
            self.pooling_net.train()
            self.rank_net.train()
            l_idx = torch.randperm(self.data.n_train)
            r_idx = torch.randperm(self.data.n_train)
            bar = trange(0, self.data.n_train, self.batch_size)
            bar.set_description_str(f'Epoch {ep}:')
            for i in bar:
                opt.zero_grad()
                l_idx_batch = l_idx[i: min(i + self.batch_size, self.data.n_train)]
                r_idx_batch = r_idx[i: min(i + self.batch_size, self.data.n_train)]
                l_batch = [self.data.x_train[j].to('cuda') for j in l_idx_batch]
                r_batch = [self.data.x_train[j].to('cuda') for j in r_idx_batch]
                y_batch = (self.data.y_train[l_idx_batch] > self.data.y_train[r_idx_batch]).float().to('cuda')
                loss = self.loss(l_batch, r_batch, y_batch, reduction='mean')
                bar.set_postfix_str(f'Batch Loss = {loss.item():.5f}')
                loss.backward()
                opt.step()

class ContrastiveLearner:
    def __init__(self, **kwargs):
        self.data = kwargs['data']
        self.loss = kwargs['contrastive_loss']
        self.history = {}
        self.reset_history()

    def reset_history(self):
        raise NotImplementedError

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def train(self, n_epochs=10001, interval=100, batch_size=128, lr=1e-4):
        self.loss.distance_net.to('cuda')
        opt = AdamW(self.loss.distance_net.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
        ep_time = []
        for ep in range(n_epochs):
            if ep % interval == 0:
                self.evaluate(ep)
            start = time.time()
            self.loss.train()
            np.random.shuffle(self.data.train_data['anchors'])
            bar = trange(0, self.data.train_data['anchors'].shape[0], batch_size)
            avg_loss = 0.
            for i in bar:
                opt.zero_grad()
                anchor_batch = self.data.train_data['anchors'][i: min(self.data.train_data['anchors'].shape[0], i + batch_size)]    
                anchor, negative, positive = self.data.create_triplet_batch(anchor_batch)                
                loss = self.loss(anchor, positive, negative)
                avg_loss += loss * anchor_batch.shape[0]
                bar.set_description_str(f'Epoch {ep} LR = {scheduler.get_last_lr()} Avg Loss = {avg_loss.item() / self.data.train_data["anchors"].shape[0]:.5f}')
                loss.backward()
                opt.step()
            scheduler.step()
            end = time.time()
            ep_time.append(end - start)
        self.history['training_time'] = ep_time
class PPI(ContrastiveLearner):
    def __init__(self, **kwargs):
        super(PPI, self).__init__(**kwargs)
        test_pairs = set()
        for i in range(len(self.data.test_data['pos'])):
            for j in self.data.test_data['pos'][i]:
                test_pairs.add((min(i, j), max(i, j), 1))
        for i in range(len(self.data.test_data['neg'])):
            for j in self.data.test_data['neg'][i]:
                test_pairs.add((min(i, j), max(i, j), 0))
        self.test_pairs = list(test_pairs)
        self.n_test = len(self.test_pairs)
    
    def reset_history(self):
        self.history = {
            'epochs': [],
            'coeffs': [],
            'acc': [],
            'dist': []
        }
    
    def evaluate(self, epoch, **kwargs):
        batch_size = kwargs.get('eval_batch_size', 128)
        with torch.no_grad():
            self.loss.eval()
            bar = trange(0, self.n_test, batch_size)    
            bar.set_description_str(f'Testing: ')
            dist, labels = [], []
            for i in bar:
                batch_pairs = self.test_pairs[i: min(i + batch_size, self.n_test)]
                batch_left, batch_right = [], []
                for l, r, label in batch_pairs:
                    batch_left.append(self.data.test_data['seq_emb'][l].to('cuda'))
                    batch_right.append(self.data.test_data['seq_emb'][r].to('cuda'))
                    labels.append(label)
                dist.append(self.loss.distance_net(batch_left, batch_right).to('cpu'))
            labels = torch.tensor(labels)
            dist = (torch.cat(dist, dim=0).flatten() + 1.) / 2.
            acc = (torch.sum((dist < 0.3) == labels) / dist.shape[0]).item()
            coeff = torch.corrcoef(torch.stack([dist, labels]))[0, 1].item()
            print(f'Coeff = {coeff:.5f} Acc = {acc:.5f}')
        self.history['epochs'].append(epoch)
        self.history['acc'].append(acc)
        self.history['coeffs'].append(coeff)
        self.history['dist'].append(dist)

class DPI(ContrastiveLearner):
    def __init__(self, **kwargs):
        super(DPI, self).__init__(**kwargs)
        self.test_pairs = {}
        for fold in TEST_DPI_DOMAINS.keys():
            test_pairs = []
            pos, neg = self.data.test_data[fold]['pos'], self.data.test_data[fold]['neg']
            for i in range(len(pos)):
                for j in pos[i]:
                    test_pairs.append((i, j, 1))
            for i in range(len(neg)):
                for j in neg[i]:
                    test_pairs.append((i, j, 0))
            self.test_pairs[fold] = test_pairs
        
    def reset_history(self):
        self.history = {
            fold: {
                'epochs': [],
                'coeffs': [],
                'acc': [],
                'dist': [],
                'auc': []
            } for fold in TEST_DPI_DOMAINS
        }
    
    def evaluate(self, epoch, **kwargs):
        batch_size = kwargs.get('eval_batch_size', 256)
        for fold in TEST_DPI_DOMAINS.keys():
            with torch.no_grad():
                self.loss.eval()
                n_test = len(self.test_pairs[fold])
                bar = trange(0, n_test, batch_size)    
                bar.set_description_str(f'Testing {fold}:')
                dist, labels = [], []
                for i in bar:
                    batch_pairs = self.test_pairs[fold][i: min(i + batch_size, n_test)]
                    batch_left, batch_right = [], []
                    for l, r, label in batch_pairs:
                        lemb = self.data.test_data[fold]['dom_emb'][l].to('cuda')
                        remb = self.data.test_data[fold]['pep_emb'][r].to('cuda')
                        batch_left.append(lemb)
                        batch_right.append(remb)
                        labels.append(label)
                    dist.append(self.loss.distance_net(batch_left, batch_right).to('cpu'))
                labels = torch.tensor(labels)
                # dist = (torch.cat(dist, dim=0).flatten() + 1.) / 2.
                dist = torch.cat(dist, dim=0).flatten()
                dist = torch.sigmoid(2.5 * dist)
                acc = (torch.sum((dist < 0.5) == labels) / dist.shape[0]).item()
                auc = roc_auc_score(labels.numpy(), 1.0 - dist.numpy())
                coeff = torch.corrcoef(torch.stack([dist, labels]))[0, 1].item()
                print(f'Coeff = {coeff:.5f} Acc = {acc:.5f} AUC = {auc:.5f}')
                self.history[fold]['epochs'].append(epoch)
                self.history[fold]['acc'].append(acc)
                self.history[fold]['coeffs'].append(coeff)
                self.history[fold]['dist'].append(dist)
                self.history[fold]['auc'].append(auc)
            

class SCOPe(ContrastiveLearner):
    def __init__(self, **kwargs):
        super(SCOPe, self).__init__(**kwargs)
        self.generate_test_pairs()

    def generate_test_pairs(self):
        test_pairs = []
        n_test = self.data.test_data.shape[0]
        for i in range(n_test):
            for j in range(i+1, n_test):
                si, sj = self.data.test_data[i], self.data.test_data[j]
                _, sfi, _, si  = self.data.train_data['meta_info']['membership'][si]
                _, sfj, _, sj  = self.data.train_data['meta_info']['membership'][sj]
                test_pairs.append((si, sj, 1 if sfi == sfj else 0))
        self.test_pairs = test_pairs
        self.n_test = len(self.test_pairs)
    
    def reset_history(self):
        self.history = {
            'epochs': [],
            'coeffs': [],
            'acc': [],
            'auc': [],
            'dist': []
        }
    
    def evaluate(self, epoch, **kwargs):
        batch_size = kwargs.get('eval_batch_size', 128)
        with torch.no_grad():
            self.loss.eval()
            bar = trange(0, self.n_test, batch_size)    
            bar.set_description_str(f'Testing:')
            dist, labels = [], []
            for i in bar:
                batch_pairs = self.test_pairs[i: min(i + batch_size, self.n_test)]
                batch_left, batch_right = [], []
                for l, r, label in batch_pairs:
                    batch_left.append(self.data.train_data['seq_emb'][l].to('cuda'))
                    batch_right.append(self.data.train_data['seq_emb'][r].to('cuda'))
                    labels.append(label)
                dist.append(self.loss.distance_net(batch_left, batch_right).to('cpu'))
            labels = torch.tensor(labels)
            dist = torch.cat(dist, dim=0).flatten()
            dist = torch.sigmoid(2.5 * dist)
            acc = (torch.sum((dist < 0.5) == labels) / dist.shape[0]).item()
            auc = roc_auc_score(labels.numpy(), 1.0 - dist.numpy())
            coeff = torch.corrcoef(torch.stack([dist, labels]))[0, 1].item()
            print(f'Coeff = {coeff:.5f} Acc = {acc:.5f} AUC = {auc:.5f}')
        self.history['epochs'].append(epoch)
        self.history['acc'].append(acc)
        self.history['auc'].append(auc)
        self.history['coeffs'].append(coeff)
        self.history['dist'].append(dist)


class SCOPev2(SCOPe):
    def __init__(self, **kwargs):
        super(SCOPev2, self).__init__(**kwargs)
    
    def generate_test_pairs(self):
        test_pairs = []
        test_idx = [
            i for i in range(len(self.data.test_adj_list)) 
            if (len(self.data.test_adj_list[i]) > 1) 
            and (len(self.data.test_adj_list[i]) < 200)
        ]
        n_test = len(test_idx)
        for i in range(n_test):
            for j in range(i+1, n_test):
                si, sj = test_idx[i], test_idx[j]
                test_pairs.append((si, sj, 1 if si in self.data.test_adj_list[sj] else 0))
        self.test_pairs = test_pairs
        self.n_test = len(self.test_pairs)

    def evaluate(self, epoch, **kwargs):
        batch_size = kwargs.get('eval_batch_size', 128)
        with torch.no_grad():
            self.loss.eval()
            bar = trange(0, self.n_test, batch_size)    
            bar.set_description_str(f'Testing:')
            dist, labels = [], []
            for i in bar:
                batch_pairs = self.test_pairs[i: min(i + batch_size, self.n_test)]
                batch_left, batch_right = [], []
                for l, r, label in batch_pairs:
                    batch_left.append(self.data.train_data['emb'][l].to('cuda'))
                    batch_right.append(self.data.train_data['emb'][r].to('cuda'))
                    labels.append(label)
                dist.append(self.loss.distance_net(batch_left, batch_right).to('cpu'))
            labels = torch.tensor(labels)
            dist = torch.cat(dist, dim=0).flatten()
            dist = torch.sigmoid(2.5 * dist)
            acc = (torch.sum((dist < 0.5) == labels) / dist.shape[0]).item()
            auc = roc_auc_score(labels.numpy(), 1.0 - dist.numpy())
            coeff = torch.corrcoef(torch.stack([dist, labels]))[0, 1].item()
            print(f'Coeff = {coeff:.5f} Acc = {acc:.5f} AUC = {auc:.5f}')
        self.history['epochs'].append(epoch)
        self.history['acc'].append(acc)
        self.history['auc'].append(auc)
        self.history['coeffs'].append(coeff)
        self.history['dist'].append(dist)