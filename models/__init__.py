from .MISA import MISA
from .DecAlign import DecAlign
from .DMD import DMD
from .DAR import DAR
from .DP import DP
from .LNLN import LNLN
from .MFM import MFM

MODEL_MAP = {
    'misa': MISA,
    'decalign': DecAlign,
    'dmd': DMD,
    'dar': DAR,
    'dp': DP,
    'lnln': LNLN,
    'mfm': MFM
}


def build_model(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError("Unsupported model name")
    return MODEL_MAP[args.model_name](args)


__all__ = ['MISA', 'DecAlign', 'DMD', 'DAR', 'DP', 'LNLN', 'MFM', 'build_model']