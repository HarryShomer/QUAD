"""
Original source can be found here -> https://github.com/migalkin/StarE/blob/master/loops/sampler.py
"""
from collections import defaultdict
from utils.utils_gcn import *
from utils.utils import *


class MultiClassSampler:
    """
        The sampler for the multi-class BCE training (instead of pointwise margin ranking)
        The output is a batch of shape (bs, num_entities)
        Each row contains 1s (or lbl-smth values) if the triple exists in the training set
        So given the triples (0, 0, 1), (0, 0, 4) the label vector will be [0, 1, 0, 0, 1]
    """
    def __init__(self, data: Union[np.array, list], n_entities: int, lbl_smooth: float = 0.0, bs: int = 64, aux_train=False, aux_lbl_smooth = 0):
        """

        :param data: data as an array of statements of STATEMENT_LEN, e.g., [0,0,0] or [0,1,0,2,4]
        :param n_entities: total number of entities
        :param lbl_smooth: whether to apply label smoothing used later in the BCE loss
        :param bs: batch size
        :param aux_train: Whether to include qual sampler
        """
        self.bs = bs
        self.data = data
        self.n_entities = n_entities
        self.lbl_smooth = lbl_smooth
        self.aux_lbl_smooth = aux_lbl_smooth
        self.aux_train = aux_train

        # Creates self.obj_index -> See `build_index` for explanation
        self.build_index()

        # Keys = training samples/stmts w/o objects
        self.obj_keys = list(self.obj_index.keys())

        self.shuffle()


    def shuffle(self):
        """
        Shuffle keys for both indices
        """
        np.random.shuffle(self.obj_keys)


    def build_index(self):
        """
        Creates following two objects:

        1. self.obj_index = defaultdict
               keys -> (s, r, quals). Ex: (11240, 556, 55, 11285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
               vals -> List of possible object entities. Ex: [2185, 6042]
        
        2. self.qual_label_index = defaultdict (see inline comments for more details)
               keys -> (s, r, o, quals, *quals*). Ex: (11240, 556, 11285, ****)
               vals -> List of possible qv entities. Ex: [2185, 6042]
        
        3. self.qual_index = defaultdict
            keys -> (s, r, quals). Ex: (11240, 556, 55, 11285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            vals -> List of possible [o, quals_wo_pair, qv, qv_ix]
        """
        self.obj_index = defaultdict(list)
        self.qual_index = defaultdict(list)
        self.qual_label_index = defaultdict(list)

        for statement in self.data:
            s, r, o, quals = statement[0], statement[1], statement[2], statement[3:] if self.data.shape[1] > 3 else None
            self.obj_index[(s, r, *quals)].append(o)

            for i in range(0, len(quals), 2):
                qr_ix, qv_ix = i, i+1

                if quals[qr_ix] != 0:
                    # For qual pair (3, 7) transform so that qr at end and append qv
                    # [2,6,3,7,4,8,0,0,0,0,0,0] -> [2,6,4,8,0,0,0,0,0,0,3] & 7
                    quals_wo_pair = np.array([quals[j] for j in range(len(quals)) if j not in [qr_ix, qv_ix]] + [quals[qr_ix]])

                    self.qual_index[(s, r, *quals)].append([o, quals_wo_pair, qv_ix])
                    self.qual_label_index[(s, r, o, *quals_wo_pair)].append(quals[qv_ix])                    

        # Remove duplicates in the objects list for convenience
        for k, v in self.obj_index.items():
            self.obj_index[k] = list(set(v))


    def reset(self, *ignore_args):
        """
        Reset the pointers of the iterators at the end of an epoch
        :return:
        """
        self.i = 0
        self.shuffle()

        return self


    def get_obj_label(self, statements):
        """
        NOTE: 1-N Training!

        Get labels for stmts

        1. For each stmt we want a label for each entity. Stored in y and initialized to 0 for all
        2. Go through each stmt
        3. Pass to index which returns list of entities where True
        4. Assign 1 for those where the entity object exists for stmt
        5. Apply smoothing

        :param statements: array of shape (bs, seq_len) like (64, 43)
        :return: array of shape (bs, num_entities) like (64, 49113)
        """
        # statement shape for correct processing of the very last batch which size might be less than self.bs
        y = np.zeros((statements.shape[0], self.n_entities), dtype=np.float32)

        for i, s in enumerate(statements):
            s, r, quals = s[0], s[1], s[2:] if self.data.shape[1] > 3 else None
            lbls = self.obj_index[(s, r, *quals)]
            y[i, lbls] = 1.0

        if self.lbl_smooth != 0.0:
            y = (1.0 - self.lbl_smooth)*y + (1.0 / self.n_entities)

        return y


    def get_qual_label(self, statements):
        """
        Same as `get_obj_label` but for qual entities

        :param statements: List. Each entry of form [s, r, o, qv, quals_wo_pair]
        :return: array of shape (?, num_entities) like (64, 49113)
        """
        # statement shape for correct processing of the very last batch which size might be less than self.bs
        y = np.zeros((statements.shape[0], self.n_entities), dtype=np.float32)

        for i, s in enumerate(statements):
            s, r, o, quals_wo_pair = s[0], s[1], s[2], s[3 + 1 + 12:]  # 3 = s, r, o | 1 = qv_ix | 12 = 6 qual pairs
            lbls = self.qual_label_index[(s, r, o, *quals_wo_pair)]
            y[i, lbls] = 1.0

        if self.lbl_smooth != 0.0:
            y = (1.0 - self.aux_lbl_smooth)*y + (1.0 / self.n_entities)

        return y


    # TODO: Merge this with labels?
    def get_qual_stmts(self, statements):
        """
        For a given list of object stmts get the appropriate qualifier statements

        :param statements: array of shape (bs, seq_len) like (64, 43)

        :return list of [s, r, o, quals, qv_ix, quals_wo_pair]
        """
        qual_stmts = []

        for stmt in statements:
            s, r, quals = stmt[0], stmt[1], stmt[2:]

            for qs in self.qual_index[tuple(stmt)]:
                o, quals_wo_pair, qv_ix = qs[0], qs[1], qs[2]
                qual_stmts.append([s, r, o, qv_ix, *quals, *quals_wo_pair])

        return np.array(qual_stmts)


    def __len__(self):
        return len(self.obj_index) // self.bs

    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        """
        Each time, take `bs` pos

        Returns:
        --------
        tuple
            _main: Original stmts -> (BS, stmt_len)
            _labels: Label for each entity and stmt -> (128, NUM_ENTITIES)
            _stmts_quals (optional): qual stmts
            _stmt_labels (optional): qual stmt labels
        """
        if self.i >= len(self.obj_keys)-1:  # otherwise batch norm will fail
            raise StopIteration

        _stmts_obj  = self.obj_keys[self.i: min(self.i + self.bs, len(self.obj_keys))]

        # Get object training samples and labels
        _main = np.array([list(x) for x in _stmts_obj])   # stmt -> list
        _obj_labels = self.get_obj_label(_main)           # Return True/False for specific entity -> 1/0...with smoothing

        # Get qual training samples and labels
        if self.aux_train:
            _stmts_quals = self.get_qual_stmts(_main)
            _stmt_labels = self.get_qual_label(_stmts_quals)

        # Increment Iterator
        self.i = min(self.i + self.bs, len(self.obj_keys))

        if self.aux_train:
            return _main, _obj_labels, _stmts_quals, _stmt_labels
        else:
            return _main, _obj_labels
