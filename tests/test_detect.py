import cv2
import numpy as np
import pytest

from cameras.detect import LedColor, classify_led, color_masks

SIZE = 40


def led(hue, s=255, v=255, radius=10, core_white=0, background=(20, 20, 20)):
    """A synthetic LED: a coloured disc on a dark box, optionally with a blown-out white core."""
    img = np.full((SIZE, SIZE, 3), background, dtype=np.uint8)
    hsv = np.uint8([[[hue, s, v]]])
    bgr = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(img, (SIZE // 2, SIZE // 2), radius, bgr, -1)
    if core_white:
        cv2.circle(img, (SIZE // 2, SIZE // 2), core_white, (255, 255, 255), -1)
    return img


@pytest.mark.parametrize("hue,expected", [
    (60, LedColor.GREEN),
    (15, LedColor.AMBER),
    (3, LedColor.RED),
    (175, LedColor.RED),   # red wraps around the end of the hue scale
])
def test_colours(hue, expected):
    assert classify_led(led(hue)).color == expected


def test_dark_box_is_off():
    assert classify_led(np.full((SIZE, SIZE, 3), 20, dtype=np.uint8)).color == LedColor.OFF


def test_tiny_glint_is_off():
    assert classify_led(led(60, radius=1)).color == LedColor.OFF


def test_blown_out_core_with_amber_ring_is_amber():
    assert classify_led(led(15, radius=10, core_white=6)).color == LedColor.AMBER


def test_white_blob_is_other():
    assert classify_led(led(0, s=0, v=255)).color == LedColor.OTHER


def test_blue_is_other():
    assert classify_led(led(120)).color == LedColor.OTHER


def test_dim_saturated_led_still_counts():
    # Under v_min (not "lit") but bright and saturated enough to vote on colour.
    assert classify_led(led(60, v=140)).color == LedColor.GREEN


def test_threshold_override():
    # Narrow the green range so hue 60 is no longer green.
    assert classify_led(led(60), {"green_min": 70, "green_max": 90}).color == LedColor.OTHER


def test_empty_roi_is_off():
    assert classify_led(np.zeros((0, 0, 3), dtype=np.uint8)).color == LedColor.OFF


def test_masks_shapes():
    m = color_masks(led(3))
    assert set(m) == {"lit", "colored", "red", "amber", "green"}
    assert m["red"].shape == (SIZE, SIZE) and m["red"].any() and not m["green"].any()


# ---------------------------------------------------------------- "any" zone mode

from cameras.detect import classify, classify_any  # noqa: E402


def zone(spots):
    """A wide box (a 'switch') with several LEDs: list of hues, None = off."""
    img = np.full((40, 200, 3), 20, dtype=np.uint8)
    for i, hue in enumerate(spots):
        if hue is None:
            continue
        bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(img, (15 + i * 25, 20), 5, bgr, -1)
    return img


def test_one_red_among_many_green_majority_vs_any():
    img = zone([60, 60, 60, 3, 60, 60, 60])
    assert classify_led(img).color == LedColor.GREEN   # majority vote hides it
    assert classify_any(img).color == LedColor.RED     # zone mode catches it
    assert classify(img, mode="any").color == LedColor.RED


def test_any_amber_spot():
    assert classify_any(zone([60, None, 15, 60])).color == LedColor.AMBER


def test_any_all_green():
    assert classify_any(zone([60, 60, None, 60])).color == LedColor.GREEN


def test_any_empty_zone_is_off():
    assert classify_any(zone([None, None])).color == LedColor.OFF


def test_any_ignores_specks_below_min_spot():
    img = zone([60, 60])
    img[5, 5] = (0, 0, 255)            # a single red pixel of noise
    assert classify_any(img).color == LedColor.GREEN
    assert classify_any(img, {"min_spot_px": 1}).color == LedColor.RED
