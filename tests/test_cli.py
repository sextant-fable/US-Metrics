from __future__ import annotations

import json

import us_metrics.cli as cli


def test_cli_ulipips_smoke(monkeypatch, capsys):
    monkeypatch.setattr(cli, "compute_ulipips", lambda **_: 0.123)
    rc = cli.main(
        [
            "ulipips",
            "--ref",
            "ref.png",
            "--img",
            "img.png",
            "--ckpt",
            "tinyusfm.pth",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["metric"] == "TinyUSFM-uLPIPS"
    assert out["score"] == 0.123


def test_cli_fit_nrq_smoke(monkeypatch, capsys, tmp_path):
    def _fake_fit(**_):
        return {"Thyroid": str(tmp_path / "models" / "Thyroid.npz")}

    monkeypatch.setattr(cli, "fit_nrq_models", _fake_fit)
    rc = cli.main(
        [
            "fit-nrq",
            "--clean-root",
            str(tmp_path / "clean"),
            "--ckpt",
            "tinyusfm.pth",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["metric"] == "TinyUSFM-NRQ-fit"
    assert out["n_models"] == 1
    assert "Thyroid" in out["models"]


def test_cli_nrq_smoke(monkeypatch, capsys):
    monkeypatch.setattr(cli, "score_nrq", lambda **_: -12.5)
    rc = cli.main(
        [
            "nrq",
            "--img",
            "img.png",
            "--ckpt",
            "tinyusfm.pth",
            "--models",
            "models_dir",
            "--organ",
            "Thyroid",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["metric"] == "TinyUSFM-NRQ"
    assert out["score"] == -12.5
    assert out["organ"] == "Thyroid"
