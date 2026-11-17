import unittest

from memory_prompt import NO_NOTES, build_chat_messages


class MemoryPromptTests(unittest.TestCase):
    def test_notes_are_in_system_message_in_order(self) -> None:
        context = [
            {"transcript": "My dog is Max.", "response": "Nice name."},
            {"transcript": "I like tea.", "response": "Noted."},
        ]

        messages = build_chat_messages("What is my dog called?", context)

        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        system = messages[0]["content"]
        self.assertLess(system.index("User: My dog is Max."), system.index("User: I like tea."))
        self.assertIn("<notes>", system)
        self.assertIn("You haven't mentioned that to me yet.", system)
        self.assertEqual(messages[1]["content"], "What is my dog called?")

    def test_empty_history_is_explicit(self) -> None:
        system = build_chat_messages("Hi", [])[0]["content"]
        self.assertIn(NO_NOTES, system)

    def test_prompt_has_no_concrete_example_fact(self) -> None:
        # A concrete example ("Chicago Bulls") leaked into answers during the eval.
        system = build_chat_messages("Hi", [])[0]["content"]
        self.assertNotIn("Bulls", system)


if __name__ == "__main__":
    unittest.main()
