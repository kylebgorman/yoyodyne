"""Tests datamodule instantiation."""

import csv
import os

import pytest
from torch.utils import data as torch_data

from yoyodyne import data


class TestDatamodule:

    @pytest.fixture
    def paths(self, tmp_path):
        # Creates a temporary directory for the model.
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # Creates a temporary TSV file with content.
        tsv_path = tmp_path / "test.tsv"
        with open(tsv_path, "w", encoding="utf-8") as sink:
            tsv_writer = csv.writer(sink, delimiter="\t")
            tsv_writer.writerow(
                ["ayılmak", "ayıldık", "V;IND;1;PL;PST;POS;DECL"]
            )
        return str(model_dir), str(tsv_path)

    def test_datamodule_instantiation(self, paths):
        model_dir, tsv_path = paths
        module = data.DataModule(
            model_dir=model_dir,
            train=tsv_path,
            features_col=3,
        )
        assert module.has_target
        assert module.has_features
        assert os.path.isfile(os.path.join(model_dir, "index.pkl"))
        assert isinstance(module.train_dataloader(), torch_data.DataLoader)


@pytest.mark.parametrize(
    "row,max_source,max_features,max_target,should_raise,match",
    [
        (["abcde", "x", "y"], 3, 10, 10, True, "Source sample length"),
        (["a", "a;b;c;d", "y"], 10, 3, 10, True, "Features sample length"),
        (["a", "x", "abcde"], 10, 10, 3, True, "Target sample length"),
        (["abc", "x;y", "z"], 10, 10, 10, False, None),
    ],
)
def test_length_validation(
    tmp_path,
    row,
    max_source,
    max_features,
    max_target,
    should_raise,
    match,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tsv_path = tmp_path / "test.tsv"
    with open(tsv_path, "w", encoding="utf-8") as sink:
        tsv_writer = csv.writer(sink, delimiter="\t")
        tsv_writer.writerow(row)
    if should_raise:
        with pytest.raises(data.Error, match=match):
            data.DataModule(
                model_dir=str(model_dir),
                train=str(tsv_path),
                features_col=2,
                target_col=3,
                max_source_length=max_source,
                max_features_length=max_features,
                max_target_length=max_target,
            )
    else:
        module = data.DataModule(
            model_dir=str(model_dir),
            train=str(tsv_path),
            features_col=2,
            target_col=3,
            max_source_length=max_source,
            max_features_length=max_features,
            max_target_length=max_target,
        )
        assert module is not None
