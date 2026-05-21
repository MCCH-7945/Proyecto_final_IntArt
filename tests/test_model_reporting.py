from types import SimpleNamespace

from model_reporting import build_detection_model_report


def test_build_detection_model_report_serializes_metrics():
    metrics_box = SimpleNamespace(mp=0.8, mr=0.7, map50=0.9, map=0.55)
    report = build_detection_model_report(
        model_name="ball_model.pt",
        model_path="models/ball_detector/ball_model.pt",
        data_yaml="data/ball_dataset/data.yaml",
        split="test",
        imgsz=1280,
        conf=0.25,
        iou=0.6,
        metrics_box=metrics_box,
    )
    assert report["metrics"]["precision"] == 0.8
    assert report["metrics"]["recall"] == 0.7
    assert report["metrics"]["map50"] == 0.9
    assert report["metrics"]["map50_95"] == 0.55

