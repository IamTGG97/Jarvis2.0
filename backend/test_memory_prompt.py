import unittest

from memory_prompt import NO_NOTES, NO_VISION, build_chat_messages


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

    def test_prompt_tells_model_to_acknowledge_new_facts(self) -> None:
        system = build_chat_messages("My favorite color is green.", [])[0]["content"]
        self.assertIn("When the user tells you something about themselves", system)

    def test_empty_history_is_explicit(self) -> None:
        system = build_chat_messages("Hi", [])[0]["content"]
        self.assertIn(NO_NOTES, system)

    def test_without_vision_prompt_says_jarvis_cannot_see(self) -> None:
        system = build_chat_messages("What do you see?", [])[0]["content"]
        self.assertIn(NO_VISION, system)
        self.assertIn("I can't see anything right now.", system)

    def test_vision_context_replaces_no_vision_line(self) -> None:
        messages = build_chat_messages("Hi", [], "The object detector found: a cup.")
        system = messages[0]["content"]
        self.assertIn("<vision>\nThe object detector found: a cup.\n</vision>", system)
        self.assertNotIn(NO_VISION, system)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_prompt_has_no_concrete_example_fact(self) -> None:
        # A concrete example ("Chicago Bulls") leaked into answers during the eval.
        system = build_chat_messages("Hi", [])[0]["content"]
        self.assertNotIn("Bulls", system)


if __name__ == "__main__":
    unittest.main()
