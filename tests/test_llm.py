import unittest
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
    def test_provider_registry_mapping(self):
        self.assertIn(ProviderType.ANTHROPIC, PROVIDER_REGISTRY)
        self.assertIn(ProviderType.OPENAI, PROVIDER_REGISTRY)
        self.assertIn(ProviderType.GEMINI, PROVIDER_REGISTRY)
        self.assertIn(ProviderType.HEURISTIC, PROVIDER_REGISTRY)
        self.assertEqual(PROVIDER_REGISTRY[ProviderType.HEURISTIC], HeuristicProvider)
        self.assertEqual(REGISTRY[Provider.MOCK], Heuristic)

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
        
        # Test _extract_json / parse_json returns None on non-JSON
        invalid_raw = "This is pure unformatted plain text with no json."
        extracted = client._extract_json(invalid_raw)
        self.assertIsNone(extracted)
        self.assertIsNone(parse_json(invalid_raw))

        # Mock client.generate to return invalid text
        original_generate = client.generate
        client.generate = lambda p, s="": "Sorry, I cannot answer in JSON."
        try:
            with self.assertRaises(LLMJSONParseError) as ctx:
                client.generate_json("some prompt", raise_on_error=True)
            self.assertIn("Failed to parse valid JSON", str(ctx.exception))
            self.assertEqual(ctx.exception.raw_response, "Sorry, I cannot answer in JSON.")
            
            # When raise_on_error / strict is False, returns raw_response dictionary
            fallback = client.generate_json("some prompt", raise_on_error=False)
            self.assertIn("raw_response", fallback)

            fallback2 = client.generate_json("some prompt", strict=False)
            self.assertIn("raw_response", fallback2)

            with self.assertRaises(ParseError) as ctx2:
                client.generate_json("some prompt", strict=True)
            self.assertEqual(ctx2.exception.raw, "Sorry, I cannot answer in JSON.")
        finally:
            client.generate = original_generate

    def test_config_ergonomics_and_aliases(self):
        cfg = Config(
            provider=Provider.OPENAI,
            key="test-key",
            model="gpt-4o",
            url="https://custom.openai.com",
            temp=0.7,
            max_tokens=1024,
            timeout=15,
        )
        self.assertEqual(cfg.provider, Provider.OPENAI)
        self.assertEqual(cfg.key, "test-key")
        self.assertEqual(cfg.api_key, "test-key")
        self.assertEqual(cfg.url, "https://custom.openai.com")
        self.assertEqual(cfg.base_url, "https://custom.openai.com")
        self.assertEqual(cfg.temp, 0.7)
        self.assertEqual(cfg.temperature, 0.7)

        # Legacy initialization
        legacy_cfg = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            api_key="legacy-key",
            base_url="https://legacy.anthropic.com",
            temperature=0.5,
        )
        self.assertEqual(legacy_cfg.key, "legacy-key")
        self.assertEqual(legacy_cfg.url, "https://legacy.anthropic.com")
        self.assertEqual(legacy_cfg.temp, 0.5)

    def test_parse_json_variants(self):
        # Markdown fenced
        fenced = '```json\n{"foo": "bar"}\n```'
        self.assertEqual(parse_json(fenced), {"foo": "bar"})

        # Raw outer braces with trailing comma repair
        dirty = 'Prefix text {"foo": "bar", "nums": [1, 2, ], } suffix'
        self.assertEqual(parse_json(dirty), {"foo": "bar", "nums": [1, 2]})

    def test_client_aliases_and_subclasses(self):
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
