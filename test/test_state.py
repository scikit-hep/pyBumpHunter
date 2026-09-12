from pathlib import Path

import numpy as np
import pytest
import uproot as upr

import pyBumpHunter as BH

DATA = Path(__file__).resolve().parents[1] / "data" / "data.root"

# save_state stores flip_sig under the key "sig_flip"
KEY_TO_ATTR = {"sig_flip": "flip_sig"}
ATTR_TO_KEY = {"flip_sig": "sig_flip"}

PARAMS_1D = dict(
    rang=[0, 20],
    width_min=2,
    width_max=6,
    width_step=1,
    scan_step=1,
    npe=100,
    nworker=1,
    seed=666,
)

PARAMS_2D = dict(
    rang=[[0, 25], [0, 25]],
    width_min=[2, 2],
    width_max=[3, 3],
    width_step=[1, 1],
    scan_step=[1, 1],
    bins=[20, 20],
    npe=50,
    nworker=1,
    seed=666,
    use_sideband=True,
)

LOAD_STATE_DEFAULTS = {
    "mode": "excess",
    "rang": None,
    "bins": 60,
    "weights": None,
    "width_min": 1,
    "width_max": None,
    "width_step": 1,
    "scan_step": 1,
    "npe": 100,
    "nworker": 4,
    "seed": None,
    "use_sideband": False,
    "sigma_limit": 5,
    "str_min": 0.5,
    "str_step": 0.25,
    "str_scale": "lin",
    "signal_exp": None,
    "flip_sig": True,
    "npe_inject": 100,
    "sideband_width": None,
}

RESULT_DEFAULTS = {
    "global_Pval": 0,
    "significance": 0,
    "res_ar": [],
    "min_Pval_ar": [],
    "min_loc_ar": [],
    "min_width_ar": [],
    "t_ar": [],
    "signal_eval": 0,
    "norm_scale": None,
    "signal_min": 0,
    "signal_ratio": None,
    "data_inject": [],
}


@pytest.fixture(scope="module")
def data_bkg_1d():
    with upr.open(DATA) as f:
        data = f["data"].arrays(library="np")["data"]
        bkg = f["bkg"].arrays(library="np")["bkg"]
    return data, bkg


@pytest.fixture(scope="module")
def data_bkg_2d():
    rng = np.random.default_rng(42)
    bkg = rng.exponential(scale=[4, 4], size=(200_000, 2))
    data = np.concatenate(
        [
            rng.exponential(scale=[4, 4], size=(20_000, 2)),
            rng.multivariate_normal(
                mean=[6.0, 7.0], cov=[[3, 0.5], [0.5, 3]], size=200
            ),
        ]
    )
    return data, bkg


@pytest.fixture(scope="module")
def scanned_1d(data_bkg_1d):
    data, bkg = data_bkg_1d
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.bump_scan(data, bkg)
    return hunter


@pytest.fixture(scope="module")
def scanned_2d(data_bkg_2d):
    data, bkg = data_bkg_2d
    hunter = BH.BumpHunter2D(**PARAMS_2D)
    hunter.bump_scan(data, bkg)
    return hunter


def _same(left, right):
    if isinstance(left, np.ndarray) and left.dtype == object:
        return len(left) == len(right) and all(
            np.array_equal(a, b) for a, b in zip(left, right)
        )
    return bool(np.array_equal(left, right))


def _assert_round_trip(original, restored, state):
    for key in state:
        attr = KEY_TO_ATTR.get(key, key)
        assert _same(getattr(original, attr), getattr(restored, attr)), key


def test_save_state_covers_every_loaded_attribute(scanned_1d):
    state = scanned_1d.save_state()
    for attr in list(LOAD_STATE_DEFAULTS) + list(RESULT_DEFAULTS):
        assert ATTR_TO_KEY.get(attr, attr) in state, attr


def test_round_trip_1d(scanned_1d):
    state = scanned_1d.save_state()
    restored = BH.BumpHunter1D()
    restored.load_state(state)
    _assert_round_trip(scanned_1d, restored, state)


def test_round_trip_2d(scanned_2d):
    state = scanned_2d.save_state()
    restored = BH.BumpHunter2D()
    restored.load_state(state)
    _assert_round_trip(scanned_2d, restored, state)


def test_round_trip_keeps_result_shapes(scanned_1d):
    restored = BH.BumpHunter1D()
    restored.load_state(scanned_1d.save_state())
    for key in ("res_ar", "min_Pval_ar", "min_loc_ar", "min_width_ar", "t_ar"):
        assert np.shape(getattr(restored, key)) == np.shape(getattr(scanned_1d, key))
    assert len(restored.min_Pval_ar) == PARAMS_1D["npe"] + 1


def test_load_empty_state_restores_defaults():
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.load_state({})
    for attr, expected in LOAD_STATE_DEFAULTS.items():
        assert _same(getattr(hunter, attr), expected), attr


def test_load_empty_state_resets_results(scanned_1d):
    restored = BH.BumpHunter1D()
    restored.load_state(scanned_1d.save_state())
    restored.load_state({})
    for attr, expected in RESULT_DEFAULTS.items():
        assert _same(getattr(restored, attr), expected), attr


def test_load_empty_state_2d(scanned_2d):
    restored = BH.BumpHunter2D()
    restored.load_state(scanned_2d.save_state())
    restored.load_state({})
    for attr, expected in RESULT_DEFAULTS.items():
        assert _same(getattr(restored, attr), expected), attr
    assert set(restored.save_state()) == set(scanned_2d.save_state())


def test_flip_sig_round_trips_through_sig_flip_key(scanned_1d):
    state = scanned_1d.save_state()
    state["sig_flip"] = not scanned_1d.flip_sig
    restored = BH.BumpHunter1D()
    restored.load_state(state)
    assert restored.flip_sig == state["sig_flip"]


def test_reset_clears_results_but_keeps_parameters(data_bkg_1d):
    data, bkg = data_bkg_1d
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.bump_scan(data, bkg)
    hunter.reset()
    for attr, expected in RESULT_DEFAULTS.items():
        assert _same(getattr(hunter, attr), expected), attr
    for attr in ("rang", "width_min", "width_max", "npe", "nworker", "seed"):
        assert _same(getattr(hunter, attr), PARAMS_1D[attr]), attr
