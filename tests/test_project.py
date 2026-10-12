"""Regression checks for data isolation, packaged inference, and the upload app."""

import io
import json
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.common import ROOT, class_names
from src.predict import MAX_UPLOAD_BYTES, Predictor, decode_image
from src.train import completed_training_stages


class ImageValidationTests(unittest.TestCase):
    def test_invalid_or_empty_upload_is_rejected(self):
        for content in (b"", b"this is not an image"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                decode_image(content)

    def test_size_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "10 MB"):
            decode_image(b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_png_modes_are_converted_to_rgb(self):
        for mode in ("L", "RGBA"):
            output = io.BytesIO()
            Image.new(mode, (32, 48)).save(output, format="PNG")
            with self.subTest(mode=mode):
                image = decode_image(output.getvalue())
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (32, 48))

    def test_other_formats_are_rejected(self):
        output = io.BytesIO()
        Image.new("RGB", (32, 32)).save(output, format="GIF")
        with self.assertRaisesRegex(ValueError, "JPEG or PNG"):
            decode_image(output.getvalue())

    def test_pixel_limit_is_enforced_before_decoding(self):
        output = io.BytesIO()
        Image.new("L", (5000, 4001)).save(output, format="PNG")
        with self.assertRaisesRegex(ValueError, "20 million pixels"):
            decode_image(output.getvalue())


class DataIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "data/manifests/images.csv")

    def test_every_class_is_present_in_every_usable_split(self):
        expected = set(range(len(class_names())))
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                self.assertEqual(set(self.frame[self.frame.split == split].class_id), expected)

    def test_leaf_groups_and_duplicate_pixels_do_not_cross_splits(self):
        usable = self.frame[self.frame.split.isin(["train", "validation", "test"])]
        for key in ("group_id", "pixel_sha256"):
            with self.subTest(key=key):
                self.assertEqual(int(usable.groupby(key).split.nunique().max()), 1)


class ResumeTests(unittest.TestCase):
    def test_only_completed_stages_can_be_resumed(self):
        history = [{"stage": "head", "validation": {"macro_f1": 0.8}}]
        self.assertEqual(completed_training_stages(history, [("head", 1), ("fine_tune", 1)], 4), {"head"})
        with self.assertRaises(ValueError):
            completed_training_stages(history, [("head", 2)], 4)

    def test_early_stopped_stage_is_preserved(self):
        history = [{"stage": "head", "validation": {"macro_f1": score}} for score in (0.8, 0.7, 0.6)]
        history.append({"stage": "fine_tune", "validation": {"macro_f1": 0.9}})
        self.assertEqual(completed_training_stages(history, [("head", 5), ("fine_tune", 1)], 2), {"head", "fine_tune"})


class PackagedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(ROOT / "artifacts/model_bundle/model.pt")
        cls.sample_path = ROOT / "examples/apple_scab.jpg"
        cls.content = cls.sample_path.read_bytes()
        cls.image = decode_image(cls.content)
        cls.result = cls.predictor.predict(cls.image)

    def test_saved_bundle_contract(self):
        bundle = ROOT / "artifacts/model_bundle"
        mapping = json.loads((bundle / "class_mapping.json").read_text())
        self.assertEqual(mapping, {str(i): name for i, name in enumerate(self.predictor.names)})
        self.assertEqual(json.loads((bundle / "preprocessing.json").read_text()), self.predictor.checkpoint["preprocessing"])
        card = json.loads((bundle / "model_card.json").read_text())
        self.assertEqual(card["model_version"], self.result["model_version"])

    def test_reload_predictions_match(self):
        reloaded = Predictor(self.predictor.checkpoint_path)
        self.assertEqual(self.result, reloaded.predict(self.image))
        self.assertEqual(self.result, self.predictor.predict(self.image))

    def test_preprocessing_and_output_contract(self):
        tensor = self.predictor.transform(self.image).unsqueeze(0)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        with torch.inference_mode():
            logits = self.predictor.model(tensor)
        self.assertEqual(tuple(logits.shape), (1, 38))
        self.assertTrue(torch.isfinite(logits).all().item())
        scores = [item["score"] for item in self.result["top_three"]]
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= score <= 1 for score in scores))

    def test_upload_app_predicts_with_shared_model(self):
        with patch("streamlit.file_uploader", return_value=io.BytesIO(self.content)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
            self.assertEqual(len(app.exception), 0)
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            prediction = self.result["prediction"]
            self.assertEqual(app.subheader[0].value, f"{prediction['crop']} · {prediction['condition']}")
            self.assertEqual(app.metric[0].value, f"{prediction['score']:.1%}")
            self.assertEqual(len(app.dataframe[-1].value), 3)

    def test_upload_app_shows_unreadable_file_error(self):
        with patch("streamlit.file_uploader", return_value=io.BytesIO(b"broken upload")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 1)
            self.assertIn("valid JPEG or PNG", app.error[0].value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
