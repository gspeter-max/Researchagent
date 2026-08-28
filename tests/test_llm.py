import unittest
from agent.llm import (
    LLMClient,
    LLMConfig,
    ProviderType,
    PROVIDER_REGISTRY,
    LLMJSONParseError,
    HeuristicProvider
)


class TestLLMClient(unittest.TestCase):
    def test_provider_registry_mapping(self):
        self.assertIn(ProviderType.ANTHROPIC, PROVIDER_REGISTRY)
        self.assertIn(ProviderType.OPENAI, PROVIDER_REGISTRY)
        self.assertIn(ProviderType.GEMINI, PROVIDER_REGISTRY)
        self.assertIn(ProviderType.HEURISTIC, PROVIDER_REGISTRY)
        self.assertEqual(PROVIDER_REGISTRY[ProviderType.HEURISTIC], HeuristicProvider)

    def test_heuristic_generation(self):
        client = LLMClient(provider="heuristic")
        resp = client.generate("Hello world")
        self.assertTrue(len(resp) > 0)

    def test_heuristic_json_generation(self):
        client = LLMClient(provider="heuristic")
        data = client.generate_json("generate queries for topic: Quantum Computing")
        self.assertIsInstance(data, dict)
        self.assertIn("queries", data)

    def test_json_parse_error_raising(self):
        client = LLMClient(provider="heuristic")
        
        # Test _extract_json returns None on non-JSON
        invalid_raw = "This is pure unformatted plain text with no json."
        extracted = client._extract_json(invalid_raw)
        self.assertIsNone(extracted)

        # Mock client.generate to return invalid text
        original_generate = client.generate
        client.generate = lambda p, s="": "Sorry, I cannot answer in JSON."
        try:
            with self.assertRaises(LLMJSONParseError) as ctx:
                client.generate_json("some prompt", raise_on_error=True)
            self.assertIn("Failed to parse valid JSON", str(ctx.exception))
            self.assertEqual(ctx.exception.raw_response, "Sorry, I cannot answer in JSON.")
            
            # When raise_on_error is False, returns raw_response dictionary
            fallback = client.generate_json("some prompt", raise_on_error=False)
            self.assertIn("raw_response", fallback)
        finally:
            client.generate = original_generate


if __name__ == "__main__":
    unittest.main()
