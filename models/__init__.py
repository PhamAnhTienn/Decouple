from .MISA import MISA
from .DecAlign import DecAlign
from .DMD import DMD

MODEL_MAP = {
    'misa': MISA,
    'decalign': DecAlign,
    'dmd': DMD
}


def build_model(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError("Unsupported model name")
    return MODEL_MAP[args.model_name](args)


__all__ = ['MISA', 'DecAlign', 'DMD', 'build_model']