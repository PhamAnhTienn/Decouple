from .MISA import MISA
from .DecAlign import DecAlign
from .DMD import DMD
from .DP import DP

MODEL_MAP = {
    'misa': MISA,
    'decalign': DecAlign,
    'dmd': DMD,
    'dp': DP
}


def build_model(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError("Unsupported model name")
    return MODEL_MAP[args.model_name](args)


__all__ = ['MISA', 'DecAlign', 'DMD', 'DP', 'build_model']