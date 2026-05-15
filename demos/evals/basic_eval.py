"""pydantic_evals: dataset-driven evaluation with NO live LLM.

Concept: `pydantic_evals` runs a *task* over a `Dataset` of `Case`s and scores
each result with evaluators. Evaluators can be built-in (`IsInstance`,
`EqualsExpected`, ...) or a custom `Evaluator` subclass. `dataset.evaluate_sync`
returns an `EvaluationReport` you can `.print()`.

Goal: evaluate a deterministic task (a plain function -- no model needed) over
a small labelled dataset, mixing one built-in evaluator with one custom
evaluator, and print the resulting report. Fully offline and reproducible.

Note: current `pydantic_evals` requires an explicit `name=` for the `Dataset`.
"""

from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EqualsExpected,
    Evaluator,
    EvaluatorContext,
)


def classify_sentiment(text: str) -> str:
    """The task under evaluation: a tiny deterministic sentiment classifier.

    In a real eval this would be an agent call; here it is a pure function so
    the demo runs anywhere with no network and identical results every time.
    """
    lowered = text.lower()
    positives = {"great", "love", "excellent", "happy", "good"}
    negatives = {"terrible", "hate", "awful", "bad", "broken"}
    if any(w in lowered for w in positives):
        return "positive"
    if any(w in lowered for w in negatives):
        return "negative"
    return "neutral"


@dataclass
class NoUnknownLabel(Evaluator[str, str, None]):
    """Custom evaluator: the task must never emit an out-of-vocabulary label."""

    evaluation_name: str = "valid_label"

    def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
        return ctx.output in {"positive", "negative", "neutral"}


dataset = Dataset(
    name="sentiment-smoke-test",
    cases=[
        Case(
            name="clearly_positive",
            inputs="I love this, it is excellent!",
            expected_output="positive",
        ),
        Case(
            name="clearly_negative",
            inputs="This is terrible and broken.",
            expected_output="negative",
        ),
        Case(
            name="neutral_statement",
            inputs="The package arrived on Tuesday.",
            expected_output="neutral",
        ),
    ],
    # Dataset-level evaluators apply to every case.
    evaluators=[EqualsExpected(), NoUnknownLabel()],
)


if __name__ == "__main__":
    print("=== pydantic_evals report (no network) ===")
    report = dataset.evaluate_sync(classify_sentiment)
    report.print(include_input=True, include_output=True)

    # The report is also programmatically inspectable for CI assertions.
    # `report.averages()` is a method returning a ReportCaseAggregate (or None).
    averages = report.averages()
    print()
    print("=== Programmatic check ===")
    assert averages is not None
    assert averages.assertions == 1.0, "every case should pass all assertions"
    print("All cases passed every evaluator (assertion pass rate = 1.0)")
