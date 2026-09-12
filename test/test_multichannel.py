from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import uproot as upr

import pyBumpHunter as BH

DATA = Path(__file__).resolve().parents[1] / "data" / "data.root"

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

# Validated against the single channel case in test_BumpHunter.py
EXPECTED_LOC = 16
EXPECTED_WIDTH = 4


@pytest.fixture(autouse=True)
def _headless():
    matplotlib.use("Agg")
    yield
    plt.close("all")


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
def scanned_multi_1d(data_bkg_1d):
    data, bkg = data_bkg_1d
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.bump_scan([data, data], [bkg, bkg], multi_chan=True)
    return hunter


@pytest.fixture(scope="module")
def scanned_multi_2d(data_bkg_2d):
    data, bkg = data_bkg_2d
    hunter = BH.BumpHunter2D(**PARAMS_2D)
    hunter.bump_scan([data, data], [bkg, bkg], multi_chan=True)
    return hunter


def test_multi_chan_1d_scans_every_channel(scanned_multi_1d):
    assert np.asarray(scanned_multi_1d.res_ar).ndim == 2
    assert len(scanned_multi_1d.min_loc_ar[0]) == 2
    assert len(scanned_multi_1d.min_width_ar[0]) == 2


def test_multi_chan_1d_finds_the_known_bump(scanned_multi_1d):
    assert list(scanned_multi_1d.min_loc_ar[0]) == [EXPECTED_LOC] * 2
    assert list(scanned_multi_1d.min_width_ar[0]) == [EXPECTED_WIDTH] * 2


def test_multi_chan_1d_fills_pseudo_results(scanned_multi_1d):
    assert len(scanned_multi_1d.min_Pval_ar) == PARAMS_1D["npe"] + 1
    assert len(scanned_multi_1d.t_ar) == PARAMS_1D["npe"] + 1
    assert 0.0 <= scanned_multi_1d.global_Pval <= 1.0


def test_multi_chan_1d_plot_tomography_per_channel(
    scanned_multi_1d, data_bkg_1d, tmp_path
):
    data, _ = data_bkg_1d
    for chan in (0, 1):
        out = tmp_path / f"tomography_chan{chan}.png"
        scanned_multi_1d.plot_tomography([data, data], filename=str(out), chan=chan)
        assert out.exists() and out.stat().st_size > 0


def test_multi_chan_2d_scans_every_channel(scanned_multi_2d):
    assert np.asarray(scanned_multi_2d.res_ar).ndim == 2
    assert len(scanned_multi_2d.min_loc_ar[0]) == 2
    assert 0.0 <= scanned_multi_2d.global_Pval <= 1.0


def _scan_summary(hunter):
    return (
        hunter.global_Pval,
        float(hunter.significance),
        int(hunter.min_loc_ar[0]),
        int(hunter.min_width_ar[0]),
        float(hunter.min_Pval_ar[0]),
    )


def test_result_is_independent_of_nworker(data_bkg_1d):
    data, bkg = data_bkg_1d
    serial = BH.BumpHunter1D(**PARAMS_1D)
    serial.bump_scan(data, bkg)
    threaded = BH.BumpHunter1D(**{**PARAMS_1D, "nworker": 4})
    threaded.bump_scan(data, bkg)
    assert _scan_summary(threaded) == _scan_summary(serial)


def test_result_is_independent_of_is_hist(data_bkg_1d):
    data, bkg = data_bkg_1d
    raw = BH.BumpHunter1D(**PARAMS_1D)
    raw.bump_scan(data, bkg)
    binned = BH.BumpHunter1D(**PARAMS_1D)
    binned.bump_scan(
        np.histogram(data, bins=60, range=PARAMS_1D["rang"])[0],
        np.histogram(bkg, bins=60, range=PARAMS_1D["rang"])[0],
        is_hist=True,
    )
    assert _scan_summary(binned) == _scan_summary(raw)


def test_scan_twice_is_reproducible(data_bkg_1d):
    data, bkg = data_bkg_1d
    first = BH.BumpHunter1D(**PARAMS_1D)
    first.bump_scan(data, bkg)
    second = BH.BumpHunter1D(**PARAMS_1D)
    second.bump_scan(data, bkg)
    assert _scan_summary(second) == _scan_summary(first)


def test_deficit_mode_runs(data_bkg_1d):
    data, bkg = data_bkg_1d
    hunter = BH.BumpHunter1D(**{**PARAMS_1D, "mode": "deficit"})
    hunter.bump_scan(data, bkg)
    assert 0.0 <= hunter.global_Pval <= 1.0
    assert 0.0 < hunter.min_Pval_ar[0] <= 1.0


def test_scan_without_pseudo_data_keeps_previous_stats(data_bkg_1d):
    data, bkg = data_bkg_1d
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.bump_scan(data, bkg)
    hunter.bump_scan(data, bkg, do_pseudo=False)
    assert len(hunter.min_loc_ar) > 0
    assert int(hunter.min_loc_ar[0]) == EXPECTED_LOC
