import json

import pytest

from rg_tracer.fallback import run_academic_pipeline
from rg_tracer.fallback.bayes import Likelihood, Prior, compute_posterior


def test_academic_pipeline_outputs_bayesian_position(tmp_path):
    problem_path = tmp_path / "claims.jsonl"
    records = [
        {
            "id": "case-1",
            "claim": "Reforms lowered mortality",
            "concept": "health",
            "analysis": [
                "Reports cite a 20% decline (Lee 1920, p.5) and likely reflect reform scope.",
                "However, rural clinics contest the drop and both sides are summarised.",
            ],
            "prior": 0.6,
            "evidence": [
                {
                    "source": "Lee 1920",
                    "year": 1920,
                    "method": "archival",
                    "finding": "Urban hospitals record declines",
                    "limitations": "Urban sample",
                    "support_if_true": 0.75,
                    "support_if_false": 0.45,
                }
            ],
        },
        {
            "id": "case-2",
            "claim": "Parades guaranteed unity",
            "concept": "culture",
            "analysis": [
                "Speeches insist success without cites.",
                "Therefore unity followed.",
            ],
            "prior": 0.5,
            "evidence": [],
        },
    ]
    problem_path.write_text("\n".join(json.dumps(record) for record in records))
    output = run_academic_pipeline(problem_path)
    summary = output["summary"]
    assert summary["count"] == 2
    records = output["records"]
    confident = [record for record in records if not record["abstained"]]
    assert confident and "posterior" in confident[0]
    abstained = [record for record in records if record["abstained"]]
    assert abstained and abstained[0]["decision"] == "I don't know."
    table = confident[0]["evidence_table"]
    assert table and table[0]["source"] == "Lee 1920"
    assert confident[0]["decision_policy"] in {
        "Support with caveats",
        "Gather more evidence",
    }


def test_compute_posterior_decision_policies():
    prior = Prior(hypothesis="claim", probability=0.5)
    strong_like = Likelihood(
        evidence="compelling",
        probability_if_true=0.9,
        probability_if_false=0.1,
    )
    weak_like = Likelihood(
        evidence="mixed",
        probability_if_true=0.6,
        probability_if_false=0.5,
    )
    cautious_like = Likelihood(
        evidence="contrary",
        probability_if_true=0.2,
        probability_if_false=0.8,
    )

    confident = compute_posterior(prior, [strong_like])
    assert confident.decision_policy == "Support with caveats"

    uncertain = compute_posterior(prior, [weak_like])
    assert uncertain.decision_policy == "Gather more evidence"

    cautious = compute_posterior(prior, [cautious_like])
    assert cautious.decision_policy == "Recommend caution"


@pytest.mark.parametrize("prob", [1.5, -0.1, float("nan"), float("inf"), float("-inf")])
def test_compute_posterior_validates_prior_probability(prob):
    prior = Prior(hypothesis="claim", probability=prob)
    with pytest.raises(ValueError, match=r"prior\.probability must be a finite value"):
        compute_posterior(prior, [])


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_compute_posterior_accepts_boundary_priors(prob):
    prior = Prior(hypothesis="claim", probability=prob)
    result = compute_posterior(prior, [])
    assert result.posterior == prob


def test_conflicting_likelihoods_raise():
    prior = Prior(hypothesis="claim", probability=0.5)
    likelihoods = [
        Likelihood(evidence="forces_zero", probability_if_true=0.0, probability_if_false=0.5),
        Likelihood(evidence="forces_one", probability_if_true=0.5, probability_if_false=0.0),
    ]
    with pytest.raises(ValueError, match=r"forces posterior=0 and others force posterior=1"):
        compute_posterior(prior, likelihoods)


@pytest.mark.parametrize(
    "prob_true, prob_false",
    [
        (float("nan"), 0.5),
        (0.5, float("inf")),
        (-0.1, 0.5),
        (0.5, 1.1),
    ],
)
def test_invalid_likelihood_probabilities_raise(prob_true: float, prob_false: float):
    prior = Prior(hypothesis="claim", probability=0.5)
    likelihood = Likelihood(
        evidence="invalid", probability_if_true=prob_true, probability_if_false=prob_false
    )
    with pytest.raises(ValueError, match=r"Likelihood\."):
        compute_posterior(prior, [likelihood])


def test_forced_outcomes_respect_evidence():
    prior = Prior(hypothesis="claim", probability=0.4)
    likelihood = Likelihood(evidence="certain", probability_if_true=1.0, probability_if_false=0.0)
    result = compute_posterior(prior, [likelihood])
    assert result.posterior == 1.0
    assert result.decision_policy == "Support with caveats"
