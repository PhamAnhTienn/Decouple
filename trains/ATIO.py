import logging
from .MISA import MISA
from .DecAlign import DecAlignTrainer
from .DMD import DMD
from .DAR import DAR
from .DP import DP
from .LNLN import LNLN
from .MFM import MFM
from .CACR import CACR
__all__ = ["ATIO"]

logger = logging.getLogger('DecAlign')

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'misa': MISA,
            'decalign': DecAlignTrainer,
            'dmd': DMD,
            'dar': DAR,
            'dp': DP,
            'lnln': LNLN,
            'mfm': MFM,
            'cacr': CACR
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args.model_name](args)
