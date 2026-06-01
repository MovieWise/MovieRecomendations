from recommendation_service.ml.hybrid import HybridArtifacts, HybridInferenceService


def test_hybrid_inference_reports_missing_artifacts(tmp_path):
    service = HybridInferenceService(
        HybridArtifacts(
            ease_weights_path=str(tmp_path / "missing.npy"),
            ease_item_encoder_path=str(tmp_path / "missing.joblib"),
            ease_user_encoder_path=str(tmp_path / "missing-user.joblib"),
            ease_interactions_path=str(tmp_path / "missing.npz"),
            lgbm_ranker_path=str(tmp_path / "missing.pkl"),
        )
    )
    assert service.available is False
    assert len(service.missing_artifacts) == 3

