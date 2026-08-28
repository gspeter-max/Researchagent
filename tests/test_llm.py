"""Hermetic, table-driven unit tests for LLM strategy dispatch, JSON parsing, and fault injection."""

import unittest
from unittest.mock import patch
from agent.llm import (
    Client,
    Config,
    Provider,
    BaseProvider,
    Anthropic,
    OpenAI,
    Gemini,
    Heuristic,
    ParseError,
    ProviderError,
    LLMError,
    REGISTRY,
    parse_json,
    # Backward compatibility aliases
    LLMClient,
    LLMConfig,
    ProviderType,
    PROVIDER_REGISTRY,
    LLMJSONParseError,
    LLMProviderError,
    HeuristicProvider,
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    BaseLLMProvider,
)


class TestLLMClient(unittest.TestCase):
    """Tests LLM Strategy Dispatch, JSON Extraction Robustness, and Fault Injection."""

    def test_json_parsing_equivalence_partitions(self):
        """Table-driven test of JSON parsing permutations (pure, markdown, dirty trailing commas, non-JSON)."""
        # Arrange: Matrix of (raw_input, expected_output)
        vectors = [
            ('{"key": "value", "num": 42}', {"key": "value", "num": 42}),
            ('```json\n{"status": "ok", "items": [1, 2]}\n```', {"status": "ok", "items": [1, 2]}),
            ('```\n{"markdown_no_lang": true}\n```', {"markdown_no_lang": True}),
            ('Leading text {"score": 0.85, "tags": ["ai", "nlp", ], } trailing noise', {"score": 0.85, "tags": ["ai", "nlp"]}),
            ('Non-JSON plain text response.', None),
            ('', None),
            ('{invalid: json without quotes}', None),
        ]

        for raw_input, expected_output in vectors:
            with self.subTest(raw_input=raw_input):
                # Act
                parsed = parse_json(raw_input)

                # Assert
                self.assertEqual(parsed, expected_output)

    def test_fault_injection_json_parse_error(self):
        """Fault injection: Verifies ParseError is raised under strict mode when invalid output occurs."""
        # Arrange
        client = Client(provider=Provider.HEURISTIC)
        invalid_raw = "This is definitely not a JSON payload."

        # Act & Assert
        with patch.object(client, "generate", return_value=invalid_raw):
            # Strict mode: must raise ParseError
            with self.assertRaises(ParseError) as ctx:
                client.generate_json("test prompt", strict=True)
            self.assertEqual(ctx.exception.raw, invalid_raw)

            # Non-strict mode: returns fallback dict
            fallback = client.generate_json("test prompt", strict=False)
            self.assertEqual(fallback, {"raw_response": invalid_raw})

    def test_fault_injection_external_provider_failure_fallback(self):
        """Fault injection: Simulates Anthropic/OpenAI API failure, verifying heuristic fallback."""
        # Arrange
        cfg = Config(provider=Provider.OPENAI, key="test-key")
        client = Client(cfg=cfg)

        # Act: Inject 500 Internal Server Error in strategy
        with patch.object(client._strategy, "call", side_effect=ProviderError("openai", "HTTP 500")):
            response = client.generate("Write a research report for topic: Quantum Computing")

        # Assert: Gracefully caught error and executed heuristic fallback
        self.assertTrue(len(response) > 0)
        self.assertIn("Quantum Computing", response)

    def test_provider_registry_dispatch_invariants(self):
        """Verifies O(1) provider registry strategy mapping."""
        # Assert
        self.assertEqual(REGISTRY[Provider.ANTHROPIC], Anthropic)
        self.assertEqual(REGISTRY[Provider.OPENAI], OpenAI)
        self.assertEqual(REGISTRY[Provider.GEMINI], Gemini)
        self.assertEqual(REGISTRY[Provider.HEURISTIC], Heuristic)
        self.assertEqual(REGISTRY[Provider.MOCK], Heuristic)

    def test_backward_compatibility_aliases(self):
        # Assert
        self.assertIs(LLMClient, Client)
        self.assertIs(LLMConfig, Config)
        self.assertIs(ProviderType, Provider)
        self.assertIs(LLMJSONParseError, ParseError)
        self.assertIs(LLMProviderError, ProviderError)
        self.assertIs(AnthropicProvider, Anthropic)
        self.assertIs(OpenAIProvider, OpenAI)
        self.assertIs(GeminiProvider, Gemini)
        self.assertIs(HeuristicProvider, Heuristic)
        self.assertIs(BaseLLMProvider, BaseProvider)


if __name__ == "__main__":
    unittest.main()
