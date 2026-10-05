"""Tests for llm.py's provider switch and Gemini request/response handling (no network)."""
import os
import unittest
from unittest import mock

import llm
import parse_profiles


class ProviderTests(unittest.TestCase):
    def test_key_decides_provider(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=True):
            self.assertEqual(llm.provider(), "gemini")
            self.assertEqual(llm.model_name(), "gemini/gemini-3.5-flash-lite")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x", "GEMINI_API_KEY": "y"}, clear=True):
            self.assertEqual(llm.provider(), "anthropic")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x", "MEETKAT_PROVIDER": "gemini"}, clear=True):
            self.assertEqual(llm.provider(), "gemini")

    def test_no_key_is_fatal(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(llm.FatalLLMError):
                llm.provider()


class GeminiFormatTests(unittest.TestCase):
    def test_schema_conversion(self):
        g = llm.gemini_schema(parse_profiles.TRAIT_SCHEMA)
        self.assertEqual(g["type"], "OBJECT")
        self.assertNotIn("additionalProperties", g)
        self.assertEqual(g["properties"]["habits"]["items"]["type"], "STRING")
        self.assertIn("introvert", g["properties"]["social_energy"]["enum"])

    def test_payload_maps_roles(self):
        p = llm.gemini_payload("sys", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                       {"role": "user", "content": "c"}], {"type": "object"}, 100)
        self.assertEqual([c["role"] for c in p["contents"]], ["user", "model", "user"])
        self.assertEqual(p["systemInstruction"]["parts"][0]["text"], "sys")
        self.assertEqual(p["generationConfig"]["responseMimeType"], "application/json")

    def test_reply_text_skips_thoughts_and_detects_problems(self):
        ok = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": "{\"a\": 1}"}]}}]}
        self.assertEqual(llm.gemini_text(ok), "{\"a\": 1}")
        with self.assertRaises(llm.LLMError):
            llm.gemini_text({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]})
        with self.assertRaises(llm.LLMError):
            llm.gemini_text({"promptFeedback": {"blockReason": "SAFETY"}})


if __name__ == "__main__":
    unittest.main()
