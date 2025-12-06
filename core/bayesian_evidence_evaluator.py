"""
Bayesian Evidence Evaluator

Rigorous evidence-based belief updating using Bayes' Theorem!

Core Innovation:
- Claims start at low prior probability
- Evidence updates beliefs incrementally: P(H|E) = P(E|H) × P(H) / P(E)
- Multiple evidence pieces combine to build confidence
- Source reliability tracked over time

This makes the system SKEPTICAL and EVIDENCE-DRIVEN!
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import json
import os


class EvidenceType(Enum):
    """Types of evidence."""
    OBSERVATIONAL = "observational"  # Direct observation
    EXPERIMENTAL = "experimental"    # Controlled experiment
    MATHEMATICAL = "mathematical"    # Proof or derivation
    STATISTICAL = "statistical"      # Statistical analysis
    TESTIMONIAL = "testimonial"      # From trusted source
    ANALOGICAL = "analogical"        # By analogy to known cases


class EvidenceQuality(Enum):
    """Quality levels for evidence."""
    WEAK = 0.3      # Anecdotal, single case
    MODERATE = 0.6  # Some rigor, multiple cases
    STRONG = 0.8    # Well-designed study/proof
    VERY_STRONG = 0.95  # Replicated, rigorous


@dataclass
class Evidence:
    """A piece of evidence for or against a claim."""
    id: str
    claim_id: str  # Which claim this supports/refutes
    description: str
    evidence_type: EvidenceType
    quality: EvidenceQuality

    # Bayesian parameters
    supports: bool = True  # True = supports claim, False = refutes
    likelihood_if_true: float = 0.8  # P(E|H): probability of seeing this evidence if claim is true
    likelihood_if_false: float = 0.2  # P(E|¬H): probability of seeing this evidence if claim is false

    # Metadata
    source: str = "unknown"
    source_reliability: float = 0.7  # How reliable is this source? (0-1)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    verified: bool = False


@dataclass
class Claim:
    """A claim that can be evaluated with evidence."""
    id: str
    statement: str
    domain: str

    # Bayesian belief
    prior_probability: float = 0.1  # Start skeptical!
    posterior_probability: float = 0.1  # Updated with evidence

    # Evidence
    evidence_for: List[str] = field(default_factory=list)  # Evidence IDs
    evidence_against: List[str] = field(default_factory=list)

    # Metadata
    status: str = "unverified"  # unverified, likely, verified, rejected
    confidence: float = 0.0  # How confident are we? (based on evidence quality)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class BayesianEvidenceEvaluator:
    """
    Evaluates claims using Bayesian evidence accumulation.

    Core Principle: Claims need EVIDENCE to be believed!

    Bayesian Update:
    P(H|E) = P(E|H) × P(H) / P(E)

    Where:
    - P(H|E) = Posterior (updated belief after evidence)
    - P(E|H) = Likelihood (how likely is evidence if claim is true?)
    - P(H) = Prior (initial belief before evidence)
    - P(E) = Marginal (how likely is evidence overall?)
    """

    def __init__(
        self,
        default_prior: float = 0.1,  # Start skeptical
        acceptance_threshold: float = 0.8,  # Need 80% confidence to accept
        rejection_threshold: float = 0.2,   # Below 20% = rejected
        storage_path: str = "./bayesian_evidence.json"
    ):
        self.default_prior = default_prior
        self.acceptance_threshold = acceptance_threshold
        self.rejection_threshold = rejection_threshold
        self.storage_path = storage_path

        # Storage
        self.claims: Dict[str, Claim] = {}
        self.evidence: Dict[str, Evidence] = {}
        self.source_reliability: Dict[str, float] = {}  # Track source accuracy

        self.load()

        print("[Bayesian Evaluator] Initialized - Evidence-driven reasoning!")
        print(f"  Claims tracked: {len(self.claims)}")
        print(f"  Evidence pieces: {len(self.evidence)}")

    def add_claim(
        self,
        claim_id: str,
        statement: str,
        domain: str,
        prior: Optional[float] = None
    ) -> Claim:
        """
        Add a new claim to evaluate.

        Args:
            claim_id: Unique identifier
            statement: The claim being made
            domain: Domain (e.g., "mathematics", "physics")
            prior: Prior probability (None = use default skeptical prior)

        Returns:
            Claim object
        """
        if claim_id in self.claims:
            return self.claims[claim_id]

        prior_prob = prior if prior is not None else self.default_prior

        claim = Claim(
            id=claim_id,
            statement=statement,
            domain=domain,
            prior_probability=prior_prob,
            posterior_probability=prior_prob
        )

        self.claims[claim_id] = claim

        print(f"[Bayesian] Added claim: {statement[:60]}...")
        print(f"  Prior probability: {prior_prob:.3f} (skeptical start)")

        return claim

    def add_evidence(
        self,
        evidence_id: str,
        claim_id: str,
        description: str,
        evidence_type: EvidenceType,
        quality: EvidenceQuality,
        supports: bool = True,
        likelihood_if_true: float = 0.8,
        likelihood_if_false: float = 0.2,
        source: str = "unknown"
    ) -> Evidence:
        """
        Add evidence for or against a claim.

        This triggers Bayesian update!

        Args:
            evidence_id: Unique identifier
            claim_id: Which claim this relates to
            description: What the evidence shows
            evidence_type: Type of evidence
            quality: Quality level
            supports: True = supports claim, False = refutes
            likelihood_if_true: P(E|H) - probability of seeing this if claim is true
            likelihood_if_false: P(E|¬H) - probability of seeing this if claim is false
            source: Source of evidence

        Returns:
            Evidence object
        """
        if claim_id not in self.claims:
            raise ValueError(f"Claim {claim_id} not found. Add claim first!")

        # Get source reliability
        source_reliability = self.source_reliability.get(source, 0.7)

        # Create evidence
        evidence = Evidence(
            id=evidence_id,
            claim_id=claim_id,
            description=description,
            evidence_type=evidence_type,
            quality=quality,
            supports=supports,
            likelihood_if_true=likelihood_if_true,
            likelihood_if_false=likelihood_if_false,
            source=source,
            source_reliability=source_reliability
        )

        self.evidence[evidence_id] = evidence

        # Add to claim's evidence lists
        claim = self.claims[claim_id]
        if supports:
            claim.evidence_for.append(evidence_id)
        else:
            claim.evidence_against.append(evidence_id)

        # Update belief using Bayes' theorem!
        self._update_belief(claim_id, evidence)

        direction = "SUPPORTS" if supports else "REFUTES"
        print(f"[Bayesian] Evidence {direction} claim: {description[:50]}...")
        print(f"  Type: {evidence_type.value}, Quality: {quality.name}")
        print(f"  Updated posterior: {claim.posterior_probability:.3f}")

        return evidence

    def _update_belief(self, claim_id: str, new_evidence: Evidence):
        """
        Update belief in claim using Bayesian inference.

        Bayes' Theorem:
        P(H|E) = P(E|H) × P(H) / P(E)

        Where P(E) = P(E|H) × P(H) + P(E|¬H) × P(¬H)
        """
        claim = self.claims[claim_id]

        # Current belief (prior for this update)
        prior = claim.posterior_probability

        # Likelihoods from evidence
        if new_evidence.supports:
            # Evidence supports claim
            p_e_given_h = new_evidence.likelihood_if_true
            p_e_given_not_h = new_evidence.likelihood_if_false
        else:
            # Evidence refutes claim (flip likelihoods)
            p_e_given_h = new_evidence.likelihood_if_false
            p_e_given_not_h = new_evidence.likelihood_if_true

        # Adjust likelihoods by evidence quality and source reliability
        quality_weight = new_evidence.quality.value
        reliability = new_evidence.source_reliability
        combined_weight = quality_weight * reliability

        # Interpolate between uninformative (0.5) and full likelihood
        p_e_given_h = 0.5 + (p_e_given_h - 0.5) * combined_weight
        p_e_given_not_h = 0.5 + (p_e_given_not_h - 0.5) * combined_weight

        # Marginal probability: P(E) = P(E|H)×P(H) + P(E|¬H)×P(¬H)
        p_e = p_e_given_h * prior + p_e_given_not_h * (1 - prior)

        # Avoid division by zero
        if p_e < 1e-10:
            p_e = 1e-10

        # Posterior: P(H|E) = P(E|H) × P(H) / P(E)
        posterior = (p_e_given_h * prior) / p_e

        # Clip to valid probability range
        posterior = np.clip(posterior, 0.001, 0.999)

        # Update claim
        claim.posterior_probability = posterior
        claim.last_updated = datetime.now().isoformat()

        # Update confidence (based on amount and quality of evidence)
        self._update_confidence(claim_id)

        # Update status
        self._update_status(claim_id)

    def _update_confidence(self, claim_id: str):
        """
        Update confidence based on evidence quantity and quality.

        More high-quality evidence = higher confidence
        """
        claim = self.claims[claim_id]

        all_evidence_ids = claim.evidence_for + claim.evidence_against

        if len(all_evidence_ids) == 0:
            claim.confidence = 0.0
            return

        # Compute weighted evidence count
        total_weight = 0.0
        for eid in all_evidence_ids:
            if eid in self.evidence:
                ev = self.evidence[eid]
                weight = ev.quality.value * ev.source_reliability
                total_weight += weight

        # Confidence increases with evidence but saturates
        # Use sigmoid-like function: C = 1 - exp(-k × W)
        k = 0.3  # Controls saturation rate
        confidence = 1.0 - np.exp(-k * total_weight)

        claim.confidence = confidence

    def _update_status(self, claim_id: str):
        """Update claim status based on posterior and confidence."""
        claim = self.claims[claim_id]

        p = claim.posterior_probability
        conf = claim.confidence

        # Need both high posterior AND high confidence to verify
        if p >= self.acceptance_threshold and conf >= 0.7:
            claim.status = "verified"
        elif p >= 0.5 and conf >= 0.5:
            claim.status = "likely"
        elif p <= self.rejection_threshold:
            claim.status = "rejected"
        else:
            claim.status = "unverified"

    def evaluate_claim(self, claim_id: str) -> Dict:
        """
        Get full evaluation of a claim.

        Returns:
            Dict with posterior, confidence, status, evidence summary
        """
        if claim_id not in self.claims:
            return {'error': 'Claim not found'}

        claim = self.claims[claim_id]

        # Evidence summary
        evidence_summary = {
            'supporting': len(claim.evidence_for),
            'refuting': len(claim.evidence_against),
            'total': len(claim.evidence_for) + len(claim.evidence_against)
        }

        # Quality breakdown
        quality_counts = {q.name: 0 for q in EvidenceQuality}
        for eid in claim.evidence_for + claim.evidence_against:
            if eid in self.evidence:
                quality_counts[self.evidence[eid].quality.name] += 1

        return {
            'claim': claim.statement,
            'status': claim.status,
            'posterior_probability': claim.posterior_probability,
            'confidence': claim.confidence,
            'prior_probability': claim.prior_probability,
            'evidence': evidence_summary,
            'evidence_quality': quality_counts,
            'interpretation': self._interpret_evaluation(claim)
        }

    def _interpret_evaluation(self, claim: Claim) -> str:
        """Generate human-readable interpretation."""
        p = claim.posterior_probability
        conf = claim.confidence

        if claim.status == "verified":
            return f"VERIFIED: Strong evidence supports this claim ({p:.1%} confidence with {conf:.1%} certainty)"
        elif claim.status == "likely":
            return f"LIKELY: Evidence suggests this is probably true ({p:.1%} probability)"
        elif claim.status == "rejected":
            return f"REJECTED: Evidence contradicts this claim ({p:.1%} probability)"
        else:
            return f"UNVERIFIED: Insufficient evidence to judge ({len(claim.evidence_for + claim.evidence_against)} pieces, {conf:.1%} certainty)"

    def compare_claims(self, claim_ids: List[str]) -> List[Tuple[str, float, str]]:
        """
        Compare multiple claims by posterior probability.

        Returns:
            List of (claim_id, posterior, status) sorted by probability
        """
        results = []

        for cid in claim_ids:
            if cid in self.claims:
                claim = self.claims[cid]
                results.append((cid, claim.posterior_probability, claim.status))

        # Sort by posterior (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def update_source_reliability(self, source: str, correct: bool):
        """
        Update reliability of a source based on track record.

        Uses exponential moving average.

        Args:
            source: Source identifier
            correct: Was this source's evidence correct?
        """
        current = self.source_reliability.get(source, 0.7)

        # Update with exponential moving average
        alpha = 0.2  # Learning rate
        new_value = 1.0 if correct else 0.0
        updated = alpha * new_value + (1 - alpha) * current

        self.source_reliability[source] = updated

        print(f"[Bayesian] Source '{source}' reliability: {current:.3f} → {updated:.3f}")

    def get_most_reliable_sources(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top k most reliable sources."""
        items = list(self.source_reliability.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]

    def get_claims_needing_evidence(
        self,
        min_confidence: float = 0.7,
        max_evidence: int = 3
    ) -> List[str]:
        """
        Find claims that need more evidence.

        Args:
            min_confidence: Target confidence level
            max_evidence: Claims with fewer than this many pieces

        Returns:
            List of claim IDs needing more evidence
        """
        needing_evidence = []

        for cid, claim in self.claims.items():
            total_evidence = len(claim.evidence_for) + len(claim.evidence_against)

            if claim.confidence < min_confidence or total_evidence < max_evidence:
                needing_evidence.append(cid)

        return needing_evidence

    def get_statistics(self) -> Dict:
        """Get evaluator statistics."""
        if len(self.claims) == 0:
            return {'status': 'empty'}

        # Status counts
        status_counts = {'verified': 0, 'likely': 0, 'unverified': 0, 'rejected': 0}
        for claim in self.claims.values():
            status_counts[claim.status] += 1

        # Posterior distribution
        posteriors = [c.posterior_probability for c in self.claims.values()]
        avg_posterior = np.mean(posteriors)

        # Confidence distribution
        confidences = [c.confidence for c in self.claims.values()]
        avg_confidence = np.mean(confidences)

        return {
            'status': 'active',
            'total_claims': len(self.claims),
            'total_evidence': len(self.evidence),
            'avg_evidence_per_claim': len(self.evidence) / len(self.claims),

            # Status breakdown
            'verified_claims': status_counts['verified'],
            'likely_claims': status_counts['likely'],
            'unverified_claims': status_counts['unverified'],
            'rejected_claims': status_counts['rejected'],

            # Belief metrics
            'avg_posterior': avg_posterior,
            'avg_confidence': avg_confidence,

            # Sources
            'tracked_sources': len(self.source_reliability),
            'avg_source_reliability': np.mean(list(self.source_reliability.values())) if len(self.source_reliability) > 0 else 0
        }

    def save(self):
        """Save claims and evidence to disk."""
        try:
            data = {
                'claims': {
                    cid: {
                        'statement': c.statement,
                        'domain': c.domain,
                        'prior_probability': c.prior_probability,
                        'posterior_probability': c.posterior_probability,
                        'evidence_for': c.evidence_for,
                        'evidence_against': c.evidence_against,
                        'status': c.status,
                        'confidence': c.confidence,
                        'created_at': c.created_at,
                        'last_updated': c.last_updated
                    }
                    for cid, c in self.claims.items()
                },
                'evidence': {
                    eid: {
                        'claim_id': e.claim_id,
                        'description': e.description,
                        'evidence_type': e.evidence_type.value,
                        'quality': e.quality.name,
                        'supports': e.supports,
                        'likelihood_if_true': e.likelihood_if_true,
                        'likelihood_if_false': e.likelihood_if_false,
                        'source': e.source,
                        'source_reliability': e.source_reliability,
                        'timestamp': e.timestamp
                    }
                    for eid, e in self.evidence.items()
                },
                'source_reliability': self.source_reliability
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[Bayesian] Saved {len(self.claims)} claims, {len(self.evidence)} evidence")

        except Exception as e:
            print(f"[Bayesian] Failed to save: {e}")

    def load(self):
        """Load claims and evidence from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load claims
            for cid, cdata in data.get('claims', {}).items():
                claim = Claim(
                    id=cid,
                    statement=cdata['statement'],
                    domain=cdata['domain'],
                    prior_probability=cdata['prior_probability'],
                    posterior_probability=cdata['posterior_probability'],
                    evidence_for=cdata['evidence_for'],
                    evidence_against=cdata['evidence_against'],
                    status=cdata['status'],
                    confidence=cdata['confidence'],
                    created_at=cdata['created_at'],
                    last_updated=cdata['last_updated']
                )
                self.claims[cid] = claim

            # Load evidence
            for eid, edata in data.get('evidence', {}).items():
                evidence = Evidence(
                    id=eid,
                    claim_id=edata['claim_id'],
                    description=edata['description'],
                    evidence_type=EvidenceType(edata['evidence_type']),
                    quality=EvidenceQuality[edata['quality']],
                    supports=edata['supports'],
                    likelihood_if_true=edata['likelihood_if_true'],
                    likelihood_if_false=edata['likelihood_if_false'],
                    source=edata['source'],
                    source_reliability=edata['source_reliability'],
                    timestamp=edata['timestamp']
                )
                self.evidence[eid] = evidence

            # Load source reliability
            self.source_reliability = data.get('source_reliability', {})

            print(f"[Bayesian] Loaded {len(self.claims)} claims, {len(self.evidence)} evidence")

        except Exception as e:
            print(f"[Bayesian] Failed to load: {e}")

    def demonstrate_bayesian_update(self):
        """Demonstrate Bayesian evidence accumulation."""
        print("\n" + "="*70)
        print("BAYESIAN EVIDENCE EVALUATOR - Demonstration")
        print("="*70)

        stats = self.get_statistics()

        if stats['status'] == 'empty':
            print("\n[!] No claims tracked yet")
            return

        print(f"\n📊 STATISTICS:")
        print(f"  Total claims: {stats['total_claims']}")
        print(f"  Total evidence: {stats['total_evidence']}")
        print(f"  Avg evidence per claim: {stats['avg_evidence_per_claim']:.1f}")

        print(f"\n✅ CLAIM STATUS:")
        print(f"  Verified: {stats['verified_claims']}")
        print(f"  Likely: {stats['likely_claims']}")
        print(f"  Unverified: {stats['unverified_claims']}")
        print(f"  Rejected: {stats['rejected_claims']}")

        print(f"\n🎯 BELIEF METRICS:")
        print(f"  Average posterior: {stats['avg_posterior']:.3f}")
        print(f"  Average confidence: {stats['avg_confidence']:.3f}")

        print(f"\n📚 SOURCES:")
        print(f"  Tracked sources: {stats['tracked_sources']}")
        if stats['avg_source_reliability'] > 0:
            print(f"  Avg source reliability: {stats['avg_source_reliability']:.3f}")

        print("\n" + "="*70)


# Demo
if __name__ == "__main__":
    print("Bayesian Evidence Evaluator")
    print("Evidence-driven belief updating using Bayes' Theorem!")
    print("\nCore principle: Claims need EVIDENCE to be believed!\n")

    # Create evaluator
    evaluator = BayesianEvidenceEvaluator()

    # Example: Evaluate a mathematical claim
    claim_id = "goldbach_conjecture"
    evaluator.add_claim(
        claim_id=claim_id,
        statement="Every even integer greater than 2 is the sum of two primes",
        domain="number_theory",
        prior=0.3  # Start with some belief (it's a famous conjecture)
    )

    # Add computational evidence (supports)
    evaluator.add_evidence(
        evidence_id="comp_verification_1",
        claim_id=claim_id,
        description="Verified for all even numbers up to 4×10^18",
        evidence_type=EvidenceType.EXPERIMENTAL,
        quality=EvidenceQuality.STRONG,
        supports=True,
        likelihood_if_true=0.95,
        likelihood_if_false=0.3,
        source="computational_mathematics"
    )

    # Add theoretical evidence (supports)
    evaluator.add_evidence(
        evidence_id="theoretical_1",
        claim_id=claim_id,
        description="Chen's theorem proves every large even number is prime + semiprime",
        evidence_type=EvidenceType.MATHEMATICAL,
        quality=EvidenceQuality.VERY_STRONG,
        supports=True,
        likelihood_if_true=0.9,
        likelihood_if_false=0.1,
        source="mathematical_proof"
    )

    # Evaluate
    result = evaluator.evaluate_claim(claim_id)
    print("\n" + "="*70)
    print("EVALUATION RESULT:")
    print("="*70)
    print(f"Claim: {result['claim']}")
    print(f"Status: {result['status']}")
    print(f"Posterior: {result['posterior_probability']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Evidence: {result['evidence']['supporting']} for, {result['evidence']['refuting']} against")
    print(f"\nInterpretation: {result['interpretation']}")
    print("="*70)

    # Demonstrate
    evaluator.demonstrate_bayesian_update()
