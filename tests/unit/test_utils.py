import numpy as np
import cv2
import pytest

from ncorr_app.core import utils, NcorrImage


def test_is_real_in_bounds():
    assert utils.is_real_in_bounds(0.5, 0.0, 1.0)
    with pytest.raises(ValueError):
        utils.is_real_in_bounds(1.5, 0.0, 1.0)


def test_is_int_in_bounds():
    assert utils.is_int_in_bounds(5, 0, 10)
    with pytest.raises(ValueError):
        utils.is_int_in_bounds(11, 0, 10)


def test_is_proper_image_format_valid():
    img = np.zeros((4, 4), dtype=np.uint8)
    assert utils.is_proper_image_format(img)
    img3 = np.zeros((4, 4, 3), dtype=np.uint8)
    assert utils.is_proper_image_format(img3)


def test_is_proper_image_format_invalid():
    with pytest.raises(ValueError):
        utils.is_proper_image_format(np.zeros((4, 4, 2), dtype=np.uint8))
    with pytest.raises(TypeError):
        utils.is_proper_image_format(np.zeros((4, 4), dtype=np.int8))


def test_load_images_from_paths(tmp_path):
    img1 = np.zeros((5, 5), dtype=np.uint8)
    img2 = np.ones((5, 5), dtype=np.uint8) * 255
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    imgs = utils.load_images_from_paths([p2, p1])
    assert len(imgs) == 2
    # natural sort ensures img1 comes first
    assert imgs[0].name == "img1"
    assert imgs[1].name == "img2"


def test_ncorrimage_reduce():
    arr = (np.random.rand(10, 10) * 255).astype(np.uint8)
    img = NcorrImage(arr)
    red = img.reduce(1)
    assert red.get_gs().shape == (5, 5)
