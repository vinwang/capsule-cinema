import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from custom_tools.video_generation import minimax_h3_video_generator_tool as h3
from custom_tools.video_generation import video_generation_tool as universal


class MiniMaxH3ResolutionTest(unittest.TestCase):
    def test_custom_channel_payload_preserves_resolution(self):
        for resolution in ("480p", "768P", "1080p", "2K"):
            with self.subTest(resolution=resolution):
                payload = h3.build_minimax_h3_payload(
                    prompt="test", resolution=resolution, base_url="https://proxy.example/v2"
                )
                self.assertEqual(payload["resolution"], resolution)

    def test_default_is_480p_on_custom_channel(self):
        self.assertEqual(h3.MiniMaxH3VideoSchema(prompt="test").resolution, "480p")
        payload = h3.build_minimax_h3_payload(prompt="test", base_url="https://proxy.example")
        self.assertEqual(payload["resolution"], "480p")

    def test_official_model_resolution_limits(self):
        for base_url in ("https://api.minimaxi.com", "https://api.minimax.io"):
            for resolution in ("768P", "2K"):
                payload = h3.build_minimax_h3_payload(prompt="test", resolution=resolution, base_url=base_url)
                self.assertEqual(payload["resolution"], resolution)
            with self.assertRaisesRegex(ValueError, "768P.*2K"):
                h3.build_minimax_h3_payload(prompt="test", resolution="480p", base_url=base_url)

    def test_schema_keeps_resolution_for_all_generation_entry_points(self):
        for schema in (universal.GenerateVideoFromTextSchema, universal.GenerateVideoFromImageSchema,
                       universal.GenerateAllVideosSchema, universal.UniversalVideoGenerationSchema):
            with self.subTest(schema=schema.__name__):
                self.assertIn("resolution", schema.model_fields)

    def test_actual_http_payload_through_universal_tool(self):
        env = {"MINIMAX_API_KEY": "test-key", "MINIMAX_VIDEO_BASE_URL": "https://proxy.example",
               "MINIMAX_VIDEO_MODEL": "MiniMax-H3"}
        for mode in ("text_to_video", "image_to_video", "first_last_frame", "multimodal"):
            for resolution in ("480p", "768P", "1080p", "2K", None):
                with self.subTest(mode=mode, resolution=resolution), patch.dict(os.environ, env), \
                     patch.object(h3, "requests") as http, patch.object(h3, "_download"):
                    http.post.return_value.status_code = 200
                    http.post.return_value.json.return_value = {"task_id": "test-task"}
                    http.get.return_value.status_code = 200
                    http.get.return_value.json.return_value = {
                        "task": {"status": "succeeded", "content": {"url": "https://example.test/result.mp4"}}
                    }
                    media = {"image_path": "https://example.test/first.png"} if mode == "image_to_video" else {}
                    if mode == "first_last_frame":
                        media = {"start_image_path": "https://example.test/first.png",
                                 "end_image_path": "https://example.test/last.png"}
                    elif mode == "multimodal":
                        media = {"image_paths": ["https://example.test/ref.png"],
                                 "video_paths": ["https://example.test/ref.mp4"]}
                    args = dict(prompt="test", output_dir=h3.DEFAULT_OUTPUT_DIR, engine="minimax-h3",
                                generation_type=mode, resolution=resolution, **media)
                    if mode != "multimodal":
                        args = universal.UniversalVideoGenerationSchema(**args).model_dump()
                    result = universal.UniversalVideoGenerationTool()._run(**args)
                    self.assertIn("output_path", result, result)
                    self.assertEqual(http.post.call_count, 1)
                    self.assertEqual(http.post.call_args.kwargs["json"]["resolution"], resolution or "480p")

    def test_invalid_resolution_never_submits_or_upgrades(self):
        for resolution, base_url in (("720p", "https://proxy.example"), ("480p", "https://api.minimaxi.com")):
            with self.subTest(resolution=resolution, base_url=base_url), \
                 patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "MINIMAX_VIDEO_BASE_URL": base_url,
                                         "MINIMAX_VIDEO_MODEL": "MiniMax-H3"}), patch.object(h3, "requests") as http:
                result = h3.MiniMaxH3VideoGeneratorTool()._run(prompt="test", resolution=resolution)
                self.assertFalse(result["success"])
                http.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
