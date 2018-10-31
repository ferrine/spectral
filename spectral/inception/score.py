"""
Taken from https://github.com/mseitzer/pytorch-fid/blob/master/inception.py
"""
import functools
import numpy as np
import tensorflow as tf
import itertools
import warnings
import pathlib
import cv2
import os
from tensorflow.python.ops import array_ops, functional_ops
from scipy import linalg


tfgan = tf.contrib.gan

INCEPTION_URL = (
    "http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz"
)
INCEPTION_FROZEN_GRAPH = "inceptionv1_for_inception_score.pb"


def _default_graph_def_fn():
    os.makedirs(os.path.join(os.path.expanduser("~"), ".inception"), exist_ok=True)
    return tfgan.eval.get_graph_def_from_url_tarball(
        INCEPTION_URL,
        INCEPTION_FROZEN_GRAPH,
        os.path.join(
            os.path.expanduser("~"), ".inception", os.path.basename(INCEPTION_URL)
        ),
    )


def to_batch(iterable, n):
    if n == -1:
        yield iter(iterable)
    else:
        sourceiter = iter(iterable)
        while True:
            batchiter = itertools.islice(sourceiter, n)
            try:
                yield itertools.chain([next(batchiter)], batchiter)
            except StopIteration:
                # exhausted
                return


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
_inception_session = tf.Session(config=config)

BATCH_SIZE = 64
USE_TQDM = False
# Run images through Inception.

inception_images = tf.placeholder(tf.float32, [None, 3, None, None])


def create_inception_logits_graph(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])  # to BHWC
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    logits = functional_ops.map_fn(
        fn=functools.partial(
            tfgan.eval.run_inception,
            output_tensor="logits:0",
            default_graph_def_fn=_default_graph_def_fn,
        ),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name="RunClassifierLogits",
    )
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits


def create_inception_2048_pool_graph(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])  # to BHWC
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    final_pool = functional_ops.map_fn(
        fn=functools.partial(
            tfgan.eval.run_inception,
            output_tensor="pool_3:0",
            default_graph_def_fn=_default_graph_def_fn,
        ),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name="RunClassifierPool",
    )
    final_pool = array_ops.concat(array_ops.unstack(final_pool), 0)
    return final_pool


inception_logits = create_inception_logits_graph()
inception_final_pool = create_inception_2048_pool_graph()


def _get_inception_probs(inps):
    if isinstance(inps, str):
        if inps.endswith(".npy"):
            return np.load(inps).copy()
        else:
            raise IOError(inps)
    preds = []
    batch_iter = to_batch(iter(inps), BATCH_SIZE)
    with _inception_session.as_default():
        for batch in batch_iter:
            inp = np.asarray(list(map(_prepare_image_for_inception, batch)))
            pred = inception_logits.eval({inception_images: inp})[:, :1000]
            preds.append(pred)
    preds = np.concatenate(preds, 0)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds


def _get_inception_pooling_mu_cov(inps):
    final_pool = []
    batch_iter = to_batch(iter(inps), BATCH_SIZE)
    with _inception_session.as_default():
        for batch in batch_iter:
            inp = np.asarray(list(map(_prepare_image_for_inception, batch)))
            pred = inception_final_pool.eval({inception_images: inp})
            final_pool.append(pred)
    final_pool = np.concatenate(final_pool, 0)
    return np.mean(final_pool, axis=0), np.cov(final_pool, rowvar=False)


def kl(p, q):
    return np.sum(p * (np.log(p) - np.log(q)), -1)


def preds2score(preds, splits):
    scores = []
    for i in range(splits):
        part = preds[
            (i * preds.shape[0] // splits) : ((i + 1) * preds.shape[0] // splits), :
        ]
        kl_ = np.mean(kl(part, np.mean(part, 0, keepdims=True)))
        scores.append(np.exp(kl_))
    return np.mean(scores), np.std(scores)


def _prepare_image_for_inception(images):
    images = np.asarray(images)
    assert len(images.shape) == 3, images.shape
    assert images.shape[0] == 3, images.shape[0]
    assert np.max(images[0]) <= 1, np.max(images[0])
    assert np.min(images[0]) >= 0, np.min(images[0])
    return (images - 0.5) * 2.0


def calculate_reconstruction_inception_score(orig, recon, splits=10, kind="pq"):
    orig_probs = _get_inception_probs(orig)
    recon_probs = _get_inception_probs(recon)
    scores = []
    kind = kind.lower().replace(" ", "")
    for i in range(splits):
        part_recon = recon_probs[
            (i * recon_probs.shape[0] // splits) : (
                (i + 1) * recon_probs.shape[0] // splits
            ),
            :,
        ]
        part_orig = orig_probs[
            (i * orig_probs.shape[0] // splits) : (
                (i + 1) * orig_probs.shape[0] // splits
            ),
            :,
        ]
        if kind == "pq":
            dist = np.mean(kl(part_orig, part_recon))
        elif kind == "qp":
            dist = np.mean(kl(part_recon, part_orig))
        elif kind in {"(qp+pq)/2", "(pq+qp)/2"}:
            dist = np.mean(kl(part_recon, part_orig) + kl(part_orig, part_recon)) / 2
        elif kind == "jsd":
            part_median = (part_orig + part_recon) / 2
            dist = np.mean(kl(part_orig, part_median) + kl(part_recon, part_median)) / 2
        else:
            raise NotImplementedError(kind)
        scores.append(np.exp(dist))
    return np.mean(scores), np.std(scores)


def calculate_inception_score(images, splits=10):
    preds = _get_inception_probs(load_images(images))
    mean, std = preds2score(preds, splits)
    # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).
    return mean, std


def calculate_frechet_distance(images1, images2):
    m1, c1 = get_inception_pooling_mu_cov(images1)
    m2, c2 = get_inception_pooling_mu_cov(images2)
    return calculate_frechet_distance_from_mu_cov(m1, c1, m2, c2)


def load_images(imgs):
    """

    Parameters
    ----------
    imgs : images in BCHW format or path to images

    Returns
    -------
    np.ndarray
        prepared images for this module
    """
    if isinstance(imgs, str):
        path = pathlib.Path(imgs)
        files = list(path.glob("*.jpg")) + list(path.glob("*.png"))

        imgs = np.array([cv2.imread(str(fn)).astype(np.float32) for fn in files])

        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))

        # Rescale images to be between 0 and 1
        imgs /= 255
    return imgs


def get_inception_pooling_mu_cov(imgs):
    if isinstance(imgs, str):
        path = imgs
        if path.endswith(".npz"):
            f = np.load(path)
            m, s = f["mean"][:], f["cov"][:]
            f.close()
            return m, s
        else:
            raise IOError(imgs)
    else:
        return _get_inception_pooling_mu_cov(load_images(imgs))


def calculate_frechet_distance_from_mu_cov(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
