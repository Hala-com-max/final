import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.textgen_metric_registry import textgen_metric
from a4s_eval.service.functional_model import TextGenerationModel

from textattack import Attack, Attacker, AttackArgs
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.datasets import Dataset as TextAttackDataset
from textattack.goal_functions import UntargetedClassification
from textattack.models.wrappers import ModelWrapper
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)


METRIC_NAME = "strong_deepwordbug_char_perturbation_rate"

AG_NEWS_LABELS = {
    "0": "WORLD",
    "1": "SPORTS",
    "2": "BUSINESS",
    "3": "SCIENCE_TECHNOLOGY",
}


@dataclass(frozen=True)
class StrongDeepWordBugConfig:
    text_column: str = "question"
    label_column: str = "label"
    sample_size: int = 100
    query_budget: int = 50_000
    max_edit_distance: int = 1
    attack_examples_path: str = "./tests/data/measures/attack_examples.csv"


class A4SClassificationWrapper(ModelWrapper):
    def __init__(self, functional_model: TextGenerationModel, label_map: dict[int, str]):
        self.model = functional_model
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.label_index_map = {label: idx for idx, label in label_map.items()}

        if self.num_classes < 2:
            raise ValueError("TextAttack classification requires at least two labels.")

    def _create_prompt(self, text: str) -> str:
        allowed_labels = ", ".join(self.label_map.values())

        return (
            "Classify the news article into exactly one of these labels:\n"
            f"{allowed_labels}\n\n"
            "Rules:\n"
            "- Return only one label.\n"
            "- Do not explain.\n"
            "- Do not add punctuation.\n\n"
            f"Article:\n{text}\n\n"
            "Label:"
        )

    def __call__(self, text_list: list[str]) -> np.ndarray:
        outputs: list[np.ndarray] = []

        for text in text_list:
            scores = np.full(self.num_classes, 1.0 / self.num_classes, dtype=float)

            try:
                raw_prediction = self.model.generate_text(self._create_prompt(text))
                predicted_label = raw_prediction.strip().upper().replace(" ", "_")

                if predicted_label in self.label_index_map:
                    predicted_index = self.label_index_map[predicted_label]
                    scores[:] = 0.01 / (self.num_classes - 1)
                    scores[predicted_index] = 0.99

            except Exception:
                pass

            outputs.append(scores)

        return np.asarray(outputs)


def _validate_input_dataset(data: pd.DataFrame, config: StrongDeepWordBugConfig) -> None:
    required_columns = {config.text_column, config.label_column}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Dataset contains no rows.")


def _prepare_sample(data: pd.DataFrame, config: StrongDeepWordBugConfig) -> pd.DataFrame:
    _validate_input_dataset(data, config)

    sample_size = min(config.sample_size, len(data))

    sample_df = data[[config.text_column, config.label_column]].head(sample_size).copy()
    sample_df[config.text_column] = sample_df[config.text_column].fillna("").astype(str)

    sample_df[config.label_column] = (
        sample_df[config.label_column]
        .fillna("")
        .astype(str)
        .map(AG_NEWS_LABELS)
    )

    if sample_df[config.label_column].isna().any():
        raise ValueError("Dataset contains labels outside AG News label set {0,1,2,3}.")

    if sample_df[config.label_column].nunique() < 2:
        raise ValueError("Dataset must contain at least two unique labels.")

    return sample_df


def _build_label_maps(
    sample_df: pd.DataFrame,
    config: StrongDeepWordBugConfig,
) -> tuple[dict[int, str], dict[str, int]]:
    labels = sorted(sample_df[config.label_column].unique())
    label_map = {idx: label for idx, label in enumerate(labels)}
    label_to_index = {label: idx for idx, label in label_map.items()}

    return label_map, label_to_index


def _build_textattack_dataset(
    sample_df: pd.DataFrame,
    config: StrongDeepWordBugConfig,
    label_to_index: dict[str, int],
) -> TextAttackDataset:
    return TextAttackDataset(
        [
            (
                row[config.text_column],
                label_to_index[row[config.label_column]],
            )
            for _, row in sample_df.iterrows()
        ]
    )


def _build_attack(
    wrapper: A4SClassificationWrapper,
    config: StrongDeepWordBugConfig,
) -> Attack:
    goal = UntargetedClassification(wrapper)

    transformation = CompositeTransformation(
        [
            WordSwapRandomCharacterDeletion(random_one=True),
            WordSwapRandomCharacterInsertion(random_one=True),
            WordSwapRandomCharacterSubstitution(random_one=True),
            WordSwapNeighboringCharacterSwap(random_one=True),
        ]
    )

    constraints = [
        RepeatModification(),
        StopwordModification(),
        LevenshteinEditDistance(max_edit_distance=config.max_edit_distance),
    ]

    return Attack(goal, constraints, transformation, GreedyWordSwapWIR())


def _compute_attack_metrics(
    results: list,
    config: StrongDeepWordBugConfig,
) -> tuple[float, float, int, int]:
    perturbation_rates: list[float] = []
    attack_examples: list[dict] = []

    successful_attacks = 0
    evaluated_examples = len(results)

    for result in results:
        if not isinstance(result, SuccessfulAttackResult):
            continue

        successful_attacks += 1

        original_text = result.original_text()
        perturbed_text = result.perturbed_text()

        attack_examples.append(
            {
                "original": original_text,
                "perturbed": perturbed_text,
                "successful": True,
            }
        )

        if len(original_text) > 0:
            perturbation_rates.append(
                levenshtein_distance(original_text, perturbed_text) / len(original_text)
            )

    pd.DataFrame(attack_examples).to_csv(config.attack_examples_path, index=False)

    average_perturbation_rate = (
        sum(perturbation_rates) / len(perturbation_rates)
        if perturbation_rates
        else 0.0
    )

    success_rate = (
        successful_attacks / evaluated_examples
        if evaluated_examples > 0
        else 0.0
    )

    return average_perturbation_rate, success_rate, successful_attacks, evaluated_examples


@textgen_metric(name=METRIC_NAME)
def strong_deepwordbug_metric(
    datashape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TextGenerationModel,
) -> list[Measure]:
    if test_dataset.data is None:
        raise ValueError("Dataset is missing data for evaluation.")

    config = StrongDeepWordBugConfig()

    sample_df = _prepare_sample(test_dataset.data, config)
    label_map, label_to_index = _build_label_maps(sample_df, config)

    wrapper = A4SClassificationWrapper(functional_model, label_map)

    ta_data = _build_textattack_dataset(
        sample_df=sample_df,
        config=config,
        label_to_index=label_to_index,
    )

    attack = _build_attack(wrapper, config)

    attack_args = AttackArgs(
        num_examples=len(sample_df),
        disable_stdout=True,
        query_budget=config.query_budget,
    )

    attacker = Attacker(attack, ta_data, attack_args)

    start = time.perf_counter()
    results = attacker.attack_dataset()
    elapsed = time.perf_counter() - start

    avg_perturbation_rate, success_rate, successful_attacks, evaluated_examples = (
        _compute_attack_metrics(results, config)
    )

    return [
        Measure(
            name=METRIC_NAME,
            score=avg_perturbation_rate,
            unit="ratio",
            time=elapsed,
            index=evaluated_examples,
            meta={
                "attack_backend": "textattack",
                "attack": "deepwordbug_style_character_attack",
                "dataset": "huggingface_ag_news",
                "labels": list(label_map.values()),
                "text_column": config.text_column,
                "label_column": config.label_column,
                "sample_size": len(sample_df),
                "query_budget": config.query_budget,
                "max_edit_distance": config.max_edit_distance,
                "successful_attacks": successful_attacks,
                "evaluated_examples": evaluated_examples,
                "success_rate": success_rate,
                "attack_examples_path": config.attack_examples_path,
            },
        )
    ]