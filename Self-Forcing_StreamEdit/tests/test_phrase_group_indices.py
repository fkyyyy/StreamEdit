import unittest

from tests._pipeline_imports import load_pipeline_module


utils = load_pipeline_module("utils")
find_phrase_group_token_indices = utils.find_phrase_group_token_indices


class WhitespaceTokenizer:
    pad_token_id = None
    eos_token_id = None

    def __call__(self, text, **kwargs):
        vocabulary = {
            token: index + 1
            for index, token in enumerate(
                "a bottle with dark violet screw cap and white body pan"
                .split()
            )
        }
        return {
            "input_ids": [
                vocabulary.get(token, 999) for token in text.split()
            ]
        }


class PhraseGroupIndicesTest(unittest.TestCase):
    def test_collects_multiple_phrase_spans_from_same_prompt(self):
        result = find_phrase_group_token_indices(
            WhitespaceTokenizer(),
            "a bottle with dark violet screw cap and white body pan",
            ["dark violet", "screw cap"],
        )
        self.assertEqual(result, [[3, 4, 5, 6]])

    def test_missing_phrase_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "was not found"):
            find_phrase_group_token_indices(
                WhitespaceTokenizer(),
                "a bottle with dark violet screw cap and white body pan",
                ["countertop"],
            )


if __name__ == "__main__":
    unittest.main()
