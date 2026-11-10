from .MISA import MISA
from .DecAlign import DecAlign
from .DMD import DMD
from .DAR import DAR
from .DP import DP
from .LNLN import LNLN
from .CENET import CENET
from .SELF_MM import SELF_MM
from .MIDAS import MIDAS

MODEL_MAP = {
    'misa': MISA,
    'decalign': DecAlign,
    'dmd': DMD,
    'dar': DAR,
    'dp': DP,
    'lnln': LNLN,
    'cenet': CENET,
    'self_mm': SELF_MM,
    'midas': MIDAS,
}


def build_model(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError("Unsupported model name")
    return MODEL_MAP[args.model_name](args)


__all__ = ['MISA', 'DecAlign', 'DMD', 'DAR', 'DP', 'LNLN', 'CENET', 'SELF_MM', 'MIDAS', 'build_model']