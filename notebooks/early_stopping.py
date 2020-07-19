class EarlyStopping(object):
    
    
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a, b: False
            
    def get_state(self):
        state = {}
        state['best'] = self.best
        state['num_bad_epochs'] = self.num_bad_epochs
        return state
    
    def load_state(self, state):
        self.best = state['best']
        self.num_bad_epochs = state['num_bad_epochs']
            
    def step(self, metrics, check=False):
        if not check:
            if self.best is None:
                self.best = metrics
                return False

            if self.is_better(metrics, self.best):
                self.num_bad_epochs = 0
                self.best = metrics
            else:
                self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            return True
        
        return False
    
    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('EarlyStopping: Mode must be either \'min\' or \'max\', received ' + mode + 'instead')
        
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)