from typing import List

from jinja2 import Template

from orz.ppo import PromptDataset


# class CustomDataset(PromptDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process_dialogue(self, dialogue: List):
#         prompt_template_jinja = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """
#         prompt_instruction_template_jinja = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# This is the problem:
# {{prompt}}
# """

#         assert len(dialogue) == 2, "dialogue must contain 2 items"

#         prompt_instruction_template = Template(prompt_instruction_template_jinja)
#         prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
#         prompt_template = Template(prompt_template_jinja)
#         if self.tokenizer.bos_token_id is None:
#             bos_token = ""
#         else:
#             bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
#         prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

#         extra = {"answer": dialogue[1]["ground_truth"]["value"]}

#         return prompt, extra


# class EvalCustomDataset(PromptDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def process_dialogue(self, dialogue: dict):
#         prompt_template_jinja = """\
# {{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
# The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
# Assistant: <think>\
# """
#         prompt_instruction_template_jinja = """\
# You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
# This is the problem:
# {{prompt}}
# """
#         assert isinstance(dialogue, dict), "dialogue must be a dict"
#         assert "prompt" in dialogue, "dialogue must contain prompt"
#         assert "final_answer" in dialogue, "dialogue must contain final_answer"
#         assert "file_name" in dialogue, "dialogue must contain file_name"

#         prompt_instruction_template = Template(prompt_instruction_template_jinja)
#         prompt_instruction = prompt_instruction_template.render(prompt=dialogue["prompt"][0]["value"])
#         prompt_template = Template(prompt_template_jinja)
#         if self.tokenizer.bos_token_id is None:
#             bos_token = ""
#         else:
#             bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
#         prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

#         extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

#         return prompt, extra


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):
        # Validate dialogue length
        assert len(dialogue) == 2, "dialogue must contain 2 items"

        # Extract the user's prompt
        user_content = dialogue[0]["value"]

        # Create the message list for apply_chat_template
        messages = [
            {"role": "user", "content": user_content}
        ]

        # Generate the prompt using the tokenizer
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,           # Return a string, not token IDs
            add_generation_prompt=True  # Append the assistant's start token
        )

        # Extract the ground truth answer
        extra = {"answer": dialogue[1]["ground_truth"]["value"]}

        return prompt, extra
    

class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        # Validate dialogue structure
        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        # Extract the user's prompt from the dialogue
        user_content = dialogue["prompt"][0]["value"]

        # Create the message list for apply_chat_template
        messages = [

            {"role": "user", "content": user_content}
        ]

        # Generate the prompt using the tokenizer
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,           # Return a string, not token IDs
            add_generation_prompt=True  # Append the assistant's start token
        )

        # Extract the extra information
        extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

        return prompt, extra