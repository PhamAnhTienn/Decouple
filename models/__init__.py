from .MISA import MISA
from .DecAlign import DecAlign
from .DMD import DMD
from .DAR import DAR
from .DP import DP
from .missingTask import TFR_NET

MODEL_MAP = {
    'misa': MISA,
    'decalign': DecAlign,
    'dmd': DMD,
    'dar': DAR,
    'dp': DP,
    'tfr_net': TFR_NET,
    'tfrnet': TFR_NET,
}

try:
    from .MULT import MULT
    MODEL_MAP['mult'] = MULT
except ImportError:
    pass

try:
    from .LNLN import LNLN
    MODEL_MAP['lnln'] = LNLN
except ImportError:
    pass

try:
    from .CENET import CENET
    MODEL_MAP['cenet'] = CENET
except ImportError:
    pass


def build_model(args):
    model_name = args.model_name.lower()
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    return MODEL_MAP[model_name](args)


__all__ = ['MISA', 'DecAlign', 'DMD', 'DAR', 'DP', 'TFR_NET', 'build_model']
