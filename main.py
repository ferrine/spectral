#!/usr/bin/env python
import exman
import spectral

parser = exman.ExParser(
    root=exman.simpleroot(__file__),
    default_config_files=['local.cfg']
)

# set defaults
parser.add_argument('--name', type=str, default='unnamed')
parser.add_argument('--wasserstein', type=bool, default=False)
parser.add_argument('--sn', type=str, choices=[
    'none',
    # the case when we do not use sn at all, our baseline
    'bug',
    # the case when we use sn as proposed in https://arxiv.org/abs/1802.05957
    'fix',
    # the case with correct implementation
])
parser.add_argument('-L', type=str, default=1,
                    help='lipschitz constraint according to the selected method of sn, '
                         'it can also be `bug`, `fix` to get L according to selected method'
                    )


def isfloat(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


parser.register_validator(
    lambda p: isfloat(p.L) or p.L in {'bug', 'fix', 'auto'},
    'L should be float or bug/fix/auto'
)

parser.register_setter(
    lambda p: setattr(p, 'L', '1') if p.L == p.sn else None
)
