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
parser.add_argument('--sn', type=str)
parser.add_argument('--sn_strict', type=bool, default=True)
parser.register_validator(lambda p: spectral.norm.validate_mode(p.sn), 'Wrong spectral norm parameter')
