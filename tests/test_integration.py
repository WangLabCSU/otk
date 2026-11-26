from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from otk.cli import cli
from otk_preprocess.workflow import run_preprocessing_workflow


@pytest.mark.parametrize("family", ["ascat", "facets", "sequenza"])
@pytest.mark.parametrize("cancer_column", ["type", "type_LUAD"])
def test_result_directory_annotation(tmp_path, family, cancer_column):
    if family == "ascat":
        pd.DataFrame({
            "sample": ["S1"], "chr": ["1"], "startpos": [65419],
            "endpos": [71585], "nMajor": [3], "nMinor": [1],
        }).to_csv(tmp_path / "batch_ascat_segments.csv", index=False)
        pd.DataFrame({"sample": ["S1"], "purity": [0.8], "ploidy": [2.2]}).to_csv(
            tmp_path / "batch_ascat_pp.csv", index=False
        )
    elif family == "facets":
        pd.DataFrame({
            "chrom": ["1"], "start": [65419], "end": [71585],
            "tcn.em": [4], "lcn.em": [1],
        }).to_csv(tmp_path / "S1_facets_segments.csv", index=False)
        pd.DataFrame({"purity": [0.8], "ploidy": [2.2]}).to_csv(
            tmp_path / "S1_facets_pp.csv", index=False
        )
    else:
        pd.DataFrame({
            "chromosome": ["1"], "start.pos": [65419], "end.pos": [71585],
            "CNt": [4], "B": [1],
        }).to_csv(tmp_path / "S1_segments.txt", sep="\t", index=False)
        pd.DataFrame({"cellularity": [0.8], "ploidy": [2.2]}).to_csv(
            tmp_path / "S1_alternative_solutions.txt", sep="\t", index=False
        )

    metadata = tmp_path / "metadata.csv"
    cancer_value = "LUAD" if cancer_column == "type" else 1
    pd.DataFrame({"sample": ["S1"], cancer_column: [cancer_value]}).to_csv(metadata, index=False)
    result = run_preprocessing_workflow(tmp_path, "hg38", metadata, 1.0)
    gene = result.loc[result["gene_id"] == "ENSG00000186092"].iloc[0]
    assert gene["segVal"] == 4
    assert gene["intersect_ratio"] == 1
    assert gene[cancer_column] == cancer_value


def test_dataframe_and_gene_level_csv(tmp_path):
    gene_data = pd.DataFrame({"sample": ["S1"], "gene_id": ["G1"], "segVal": [3], "type_BRCA": [1]})
    csv = tmp_path / "input.csv"
    gene_data.to_csv(csv, index=False)
    for source in (csv, gene_data):
        pd.testing.assert_frame_equal(run_preprocessing_workflow(source), gene_data)
        result = run_preprocessing_workflow(source, extra_info=pd.DataFrame({"sample": ["S1"], "gender": ["XY"]}))
        assert result["gender"].iloc[0] == 1
        assert result["type_BRCA"].iloc[0] == 1


@pytest.mark.parametrize("cancer_column", ["type", "type_LUAD"])
@pytest.mark.parametrize("input_kind", ["dataframe", "csv"])
@pytest.mark.parametrize("extra_kind", ["dataframe", "csv"])
def test_direct_input_rejects_cancer_metadata(tmp_path, cancer_column, input_kind, extra_kind):
    data = pd.DataFrame({"sample": ["S1"], "gene_id": ["G1"], "segVal": [3]})
    source = data
    if input_kind == "csv":
        source = tmp_path / "input.csv"
        data.to_csv(source, index=False)
    metadata = pd.DataFrame({"sample": ["S1"], cancer_column: ["LUAD" if cancer_column == "type" else 1]})
    extra = metadata
    if extra_kind == "csv":
        extra = tmp_path / "metadata.csv"
        metadata.to_csv(extra, index=False)
    with pytest.raises(ValueError, match=r"extra_info cannot contain type or type_\*"):
        run_preprocessing_workflow(source, extra_info=extra)


def test_dataframe_without_gene_id_is_not_annotated(monkeypatch):
    import otk_preprocess.workflow as workflow

    def unexpected_annotation(*args, **kwargs):
        pytest.fail("direct DataFrame input should not be annotated")

    monkeypatch.setattr(workflow, "annotate_genes", unexpected_annotation)
    data = pd.DataFrame({"sample": ["S1"], "chromosome": ["1"], "start": [1], "end": [2]})
    pd.testing.assert_frame_equal(workflow.run_preprocessing_workflow(data), data)


def test_cli_forwards_preprocessing_options(tmp_path, monkeypatch):
    import otk.preprocessing

    source = tmp_path / "input.csv"
    source.write_text("sample,gene_id,segVal\nS1,G1,3\n")
    extra = tmp_path / "extra.csv"
    extra.write_text("sample,type\nS1,LUAD\n")
    captured = {}

    def fake_predict(*args):
        captured["args"] = args
        return pd.DataFrame({"prediction": [1]})

    monkeypatch.setattr(otk.preprocessing, "predict_input", fake_predict)
    result = CliRunner().invoke(cli, [
        "predict", "--input", str(source), "--output", str(tmp_path / "out.csv"),
        "--gene-build", "hg19", "--extra-info", str(extra), "--overlap", "0.5",
    ])
    assert result.exit_code == 0, result.output
    assert captured["args"][0] == source
    assert captured["args"][3:6] == ("hg19", Path(extra), 0.5)


def test_cli_rejects_cancer_metadata_for_csv(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "xgb_new"
    model_dir.mkdir(parents=True)
    (model_dir / "best_model.pkl").touch()
    monkeypatch.setenv("OTK_MODELS_DIR", str(model_dir.parent))
    source = tmp_path / "input.csv"
    source.write_text("sample,gene_id,segVal\nS1,G1,3\n")
    extra = tmp_path / "extra.csv"
    extra.write_text("sample,type_LUAD\nS1,1\n")

    result = CliRunner().invoke(cli, [
        "predict", "--input", str(source), "--output", str(tmp_path / "out.csv"),
        "--extra-info", str(extra),
    ])
    assert result.exit_code != 0
    assert "extra_info cannot contain type or type_*" in result.output