"""pytest tests/test_lp_audit.py"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lp_audit import audit  # noqa: E402
from lp_audit import io as lio  # noqa: E402
from lp_audit.checks import (  # noqa: E402
    TopologyScorer, check_leakage, degree_bins, degree_matched_negatives, uniform_negatives,
)
from lp_audit.cli import main as cli_main  # noqa: E402
from lp_audit.examples.make_examples import N_COLS, N_ROWS, build  # noqa: E402

# scenario -> {check: expected status}. Checks not listed must be "ok".
EXPECTED = {
    "clean": {},
    "leak": {"edge_leakage": "error"},
    "mismatched_negatives": {"negative_matching": "warn"},
    "dead_columns": {"dead_candidates": "warn"},
    "weak_margin": {"model_free_baseline": "warn"},
}
CHECKS = ["edge_leakage", "negative_matching", "model_free_baseline", "dead_candidates"]


@pytest.fixture(scope="module")
def scenarios():
    return build(seed=0)


@pytest.mark.parametrize("name", EXPECTED)
def test_scenario_flags_its_own_check(scenarios, name):
    report = audit(**scenarios[name])
    for check in CHECKS:
        want = EXPECTED[name].get(check)
        got = report.status(check)
        if want is not None:
            assert got == want, f"{name}/{check}: {got}"
        elif name != "dead_columns":  # padding legitimately trips the other checks too
            assert got == "ok", f"{name}/{check} should be ok, got {got}"


def test_missing_inputs_are_unchecked_not_ok(scenarios):
    kw = dict(scenarios["clean"])
    kw.update(encoder_edges=None, eval_neg=None, train_neg=None, model_auroc=None)
    r = audit(**kw)
    assert r.status("edge_leakage") == "unchecked"
    assert r.status("negative_matching") == "unchecked"
    assert r.status("model_free_baseline") == "info"


def test_leak_check_counts_reverse_relation_and_negatives():
    held = np.array([[0, 1], [0, 1]])                      # (0,0), (1,1)
    enc = np.array([[5], [5]])
    rev_with_heldout = np.array([[1], [1]])                # (col,row) = (1,1) -> pair (1,1)
    f = check_leakage(10, held, encoder_edges=enc, encoder_edges_rev=rev_with_heldout)
    assert f.status == "error" and f.evidence["heldout_in_encoder_rev"] == 1
    f = check_leakage(10, held, encoder_edges=enc, eval_neg=held)
    assert f.status == "error" and f.evidence["eval_negatives_that_are_real_edges"] == 2


def test_undirected_leak_sees_flipped_edge():
    held = np.array([[2], [7]])
    enc = np.array([[7], [2]])
    assert check_leakage(10, held, encoder_edges=enc, undirected=True).status == "error"
    assert check_leakage(10, held, encoder_edges=enc, undirected=False).status == "ok"


def test_declared_sampler_mismatch_is_an_error(scenarios):
    kw = dict(scenarios["clean"], declared_samplers={"train": "uniform", "eval": "degree_matched"})
    assert audit(**kw).status("negative_matching") == "error"


def test_negative_samplers_respect_exclusions():
    rng = np.random.default_rng(1)
    pos = np.stack([rng.integers(0, 30, 400), rng.integers(0, 20, 400)])
    pos = np.unique(pos, axis=1)
    neg = uniform_negatives(pos, pos, 30, 20, rng)
    assert not np.isin(neg[0] * 20 + neg[1], pos[0] * 20 + pos[1]).any()
    bins = degree_bins(np.bincount(pos[1], minlength=20))
    neg, _ = degree_matched_negatives(pos, pos, 30, 20, bins, rng)
    assert not np.isin(neg[0] * 20 + neg[1], pos[0] * 20 + pos[1]).any()


# ── parity with the torch originals ───────────────────────────────────────────

def _torch_build_scorers():
    """The repo's own build_scorers if importable (needs PyG); otherwise a verbatim copy of
    training/eval_topology_baseline.py::build_scorers, so the test also runs on a laptop
    without the full training stack."""
    torch = pytest.importorskip("torch")
    try:
        from training.eval_topology_baseline import build_scorers
        return torch, build_scorers
    except Exception:
        def build_scorers(A):
            A = A.float()
            deg_m, deg_g = A.sum(dim=1), A.sum(dim=0)
            s = {"gene_degree": deg_g.unsqueeze(0).expand(A.shape[0], -1).contiguous(),
                 "pref_attach": torch.outer(deg_m, deg_g)}
            S = A @ A.T
            S.fill_diagonal_(0)
            s["common_neigh"] = S @ A
            inv_log = 1.0 / torch.log(deg_g.clamp(min=2.0))
            S_aa = A @ (A * inv_log.unsqueeze(0)).T
            S_aa.fill_diagonal_(0)
            s["adamic_adar"] = S_aa @ A
            return s
        return torch, build_scorers


def test_bipartite_scorers_match_torch(scenarios):
    torch, build_scorers = _torch_build_scorers()
    kw = scenarios["clean"]
    A = torch.zeros(N_ROWS, N_COLS)
    A[kw["train_pos"][0], kw["train_pos"][1]] = 1.0
    ref = build_scorers(A)
    pairs = np.concatenate([kw["heldout_pos"], kw["eval_neg"]], axis=1)
    mine = TopologyScorer(kw["train_pos"], N_ROWS, N_COLS).score(pairs)
    for name, M in ref.items():
        np.testing.assert_allclose(mine[name], M[pairs[0], pairs[1]].numpy(), rtol=1e-5,
                                   err_msg=name)


def test_homogeneous_scorers_match_torch_ogb_formulas():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    n = 60
    e = np.unique(np.sort(rng.integers(0, n, (2, 400)), axis=0), axis=1)
    e = e[:, e[0] != e[1]]
    A = torch.zeros(n, n)
    A[e[0], e[1]] = 1.0
    A = ((A + A.T) > 0).float()
    deg = A.sum(0)
    ref = {"node_degree": deg.unsqueeze(0).expand(n, -1), "pref_attach": torch.outer(deg, deg)}
    cn = A @ A
    cn.fill_diagonal_(0)
    aa = A @ (A * (1.0 / torch.log(deg.clamp(min=2.0))).unsqueeze(1))
    aa.fill_diagonal_(0)
    ra = A @ (A * (1.0 / deg.clamp(min=1.0)).unsqueeze(1))
    ra.fill_diagonal_(0)
    ref.update(common_neigh=cn, adamic_adar=aa, resource_alloc=ra)
    q = np.stack([rng.integers(0, n, 300), rng.integers(0, n, 300)])
    mine = TopologyScorer(e, n, n, "homogeneous").score(q)
    for name, M in ref.items():
        np.testing.assert_allclose(mine[name], M[q[0], q[1]].numpy(), rtol=1e-5, err_msg=name)


# ── io and CLI ────────────────────────────────────────────────────────────────

def test_pair_loaders_roundtrip(tmp_path):
    pairs = np.array([[0, 3, 5], [2, 1, 9]])
    np.save(tmp_path / "a.npy", pairs.T)                   # N x 2
    np.save(tmp_path / "b.npy", pairs)                     # 2 x N
    np.savetxt(tmp_path / "c.csv", pairs.T + 1, fmt="%d", delimiter=",")
    for f, base in (("a.npy", 0), ("b.npy", 0), ("c.csv", 1)):
        np.testing.assert_array_equal(lio.load_pairs(str(tmp_path / f), base), pairs)


def test_matrix_and_labeled_loaders(tmp_path):
    np.savetxt(tmp_path / "m.csv", np.array([[1, 0, 1], [0, 0, 1]]), fmt="%d", delimiter=",")
    pos, r, c = lio.load_binary_matrix(str(tmp_path / "m.csv"))
    assert (r, c) == (2, 3) and pos.shape[1] == 3
    (tmp_path / "t.txt").write_text("1 1 1\n2 3 0\n3 2 1\n")
    p, n = lio.load_labeled_pairs(str(tmp_path / "t.txt"), index_base=1)
    assert p.tolist() == [[0, 2], [0, 1]] and n.tolist() == [[1], [2]]


def test_cli_exit_codes_and_json(scenarios, tmp_path, capsys):
    for name in ("clean", "leak"):
        d = tmp_path / name
        d.mkdir()
        for k, v in scenarios[name].items():
            if isinstance(v, np.ndarray):
                np.save(d / f"{k}.npy", v.T)
        argv = ["--train-pos", str(d / "train_pos.npy"), "--heldout-pos", str(d / "heldout_pos.npy"),
                "--encoder-edges", str(d / "encoder_edges.npy"), "--eval-neg", str(d / "eval_neg.npy"),
                "--train-neg", str(d / "train_neg.npy"), "--n-rows", str(N_ROWS),
                "--n-cols", str(N_COLS), "--model-auroc", "0.9", "--json", str(d / "r.json")]
        code = cli_main(argv)
        assert code == (0 if name == "clean" else 1)
        out = json.load(open(d / "r.json"))
        assert {f["check"] for f in out["findings"]} == set(CHECKS)
        assert "Scope:" in capsys.readouterr().out


def test_cli_usage_error_is_exit_2(capsys):
    assert cli_main(["--train-pos", "x.npy"]) == 2


def test_homogeneous_degree_counts_both_endpoints():
    """A node that only ever appears as the FIRST endpoint is not dead in an undirected graph."""
    from lp_audit.checks import check_dead_columns, column_degree
    pairs = np.array([[0, 0, 1], [1, 2, 2]])
    assert column_degree(pairs, 4).tolist() == [0, 1, 2, 0]
    assert column_degree(pairs, 4, undirected=True).tolist() == [2, 2, 2, 0]
    f = check_dead_columns(4, 4, pairs, pairs[:, :1], undirected=True)
    assert f.evidence["n_dead_columns"] == 1 and f.evidence["dead_row_fraction"] == 0.25
