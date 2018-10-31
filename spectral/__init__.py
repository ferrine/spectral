try:
    import tensorflow

    del tensorflow
except ImportError:
    import warnings

    warnings.warn(
        "tensorflow load failed, you will probably fail to compute inception metrics"
    )
    del warnings
from . import nets, generator, discriminator, utils, norm, datasets, gan, logging
