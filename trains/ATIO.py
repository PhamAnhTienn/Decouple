import logging
from .MISA import MISA
from .DecAlign import DecAlignTrainer
from .DMD import DMD
from .DAR import DAR
from .DP import DP
from .TFR_NET import TFR_NET

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
            'tfr_net': TFR_NET,
            'tfrnet': TFR_NET,
        }

    def getTrain(self, args):
        model_name = args.model_name.lower()
        if model_name not in self.TRAIN_MAP:
            raise ValueError(f"Unsupported model name: {args.model_name}")
        return self.TRAIN_MAP[model_name](args)
