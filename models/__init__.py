from .MISA import MISA
from .DecAlign import DecAlign
from .DMD import DMD
from .DAR import DAR
from .DP import DP

MODEL_MAP = {
    'misa': MISA,
    'decalign': DecAlign,
    'dmd': DMD,
    'dar': DAR,
    'dp': DP
}


def build_model(args):
    model_name = args.model_name.lower()
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    return MODEL_MAP[model_name](args)


__all__ = ['MISA', 'DecAlign', 'DMD', 'DAR', 'DP', 'build_model']