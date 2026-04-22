def test_cli_imports():
    from scene_classification.cli import app

    assert app is not None
    assert {c.name for c in app.registered_commands} >= {
        "prepare-data",
        "extract-features",
        "train-tabular",
        "train-cnn",
        "evaluate",
    }
