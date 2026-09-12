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


@pytest.fixture(autouse=True)
def _headless():
    matplotlib.use("Agg")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data_bkg_sig_1d():
    with upr.open(DATA) as f:
        data = f["data"].arrays(library="np")["data"]
        bkg = f["bkg"].arrays(library="np")["bkg"]
        sig = f["sig"].arrays(library="np")["sig"]
    return data, bkg, sig


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
def scanned_1d(data_bkg_sig_1d):
    data, bkg, _ = data_bkg_sig_1d
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.bump_scan(data, bkg)
    return hunter


@pytest.fixture(scope="module")
def scanned_2d(data_bkg_2d):
    data, bkg = data_bkg_2d
    hunter = BH.BumpHunter2D(**PARAMS_2D)
    hunter.bump_scan(data, bkg)
    return hunter


@pytest.fixture(scope="module")
def injected_1d(data_bkg_sig_1d):
    _, bkg, sig = data_bkg_sig_1d
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    hunter.sigma_limit = 5
    hunter.str_min = -1
    hunter.str_scale = "log"
    hunter.signal_exp = 150
    hunter.npe_inject = 50
    hunter.signal_inject(sig, bkg)
    return hunter


def _written(path):
    return path.exists() and path.stat().st_size > 0


def test_bump_info_1d(scanned_1d, data_bkg_sig_1d):
    data, _, _ = data_bkg_sig_1d
    info = scanned_1d.bump_info(data)
    assert isinstance(info, str)
    assert "Bump edges" in info


def test_bump_info_2d(scanned_2d, data_bkg_2d):
    data, _ = data_bkg_2d
    info = scanned_2d.bump_info(data)
    assert isinstance(info, str)
    assert info


def test_plot_tomography_1d(scanned_1d, data_bkg_sig_1d, tmp_path):
    data, _, _ = data_bkg_sig_1d
    out = tmp_path / "tomography.png"
    scanned_1d.plot_tomography(data, filename=str(out))
    assert _written(out)


def test_plot_bump_1d(scanned_1d, data_bkg_sig_1d, tmp_path):
    data, bkg, _ = data_bkg_sig_1d
    out = tmp_path / "bump.png"
    scanned_1d.plot_bump(data, bkg, filename=str(out))
    assert _written(out)


def test_plot_bump_2d(scanned_2d, data_bkg_2d, tmp_path):
    data, bkg = data_bkg_2d
    out = tmp_path / "bump2d.png"
    scanned_2d.plot_bump(data, bkg, filename=str(out))
    assert _written(out)


@pytest.mark.parametrize("show_Pval", [False, True])
def test_plot_stat_1d(scanned_1d, tmp_path, show_Pval):
    out = tmp_path / f"stat_{show_Pval}.png"
    scanned_1d.plot_stat(show_Pval=show_Pval, filename=str(out))
    assert _written(out)


@pytest.mark.parametrize("show_Pval", [False, True])
def test_plot_stat_2d(scanned_2d, tmp_path, show_Pval):
    out = tmp_path / f"stat2d_{show_Pval}.png"
    scanned_2d.plot_stat(show_Pval=show_Pval, filename=str(out))
    assert _written(out)


def test_plot_inject_1d(injected_1d, tmp_path):
    lin = tmp_path / "inject.png"
    log = tmp_path / "inject_log.png"
    injected_1d.plot_inject(filename=(str(lin), str(log)))
    assert _written(lin)
    assert _written(log)


def test_signal_inject_fills_sigma_ar(injected_1d):
    assert np.asarray(injected_1d.sigma_ar).ndim == 2
    assert np.asarray(injected_1d.sigma_ar).shape[1] == 3


def test_plot_stat_before_scan_raises(tmp_path):
    # The exception type is not part of the API, only the refusal to plot
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    with pytest.raises((IndexError, ValueError, RuntimeError)):
        hunter.plot_stat(filename=str(tmp_path / "never.png"))


def test_scan_with_empty_reference_raises():
    hunter = BH.BumpHunter1D(**PARAMS_1D)
    with pytest.raises(ValueError, match="no positive bin"):
        hunter.bump_scan(np.zeros(60), np.zeros(60), is_hist=True)
