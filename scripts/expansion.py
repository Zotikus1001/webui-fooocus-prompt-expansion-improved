# Fooocus GPT2 Expansion
# Algorithm created by Lvmin Zhang at 2023, Stanford 
# Modified by power88 and GPT-4o for stable-diffusion-webui
# If used inside Fooocus, any use is permitted.
# If used outside Fooocus, only non-commercial use is permitted (CC-By NC 4.0).
# This applies to the word list, vocab, model, and algorithm.


import os
import re
import torch
import math
import random
import shutil
import gradio as gr

from pathlib import Path
from modules.scripts import basedir
from huggingface_hub import hf_hub_download
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from modules import scripts, paths_internal, errors, devices
from modules.ui_components import InputAccordion
from functools import lru_cache


# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2 ** 32
neg_inf = - 8192.0
ext_dir = Path(basedir())
fooocus_expansion_model_dir = Path(paths_internal.models_path) / "prompt_expansion"


def download_model():
    fooocus_expansion_model = fooocus_expansion_model_dir / "pytorch_model.bin"
    if not fooocus_expansion_model.exists():
        try:
            print(f'### webui-fooocus-prompt-expansion: Downloading model...')
            shutil.copytree(ext_dir / "models", fooocus_expansion_model_dir)
            hf_hub_download(repo_id='lllyasviel/misc', filename='fooocus_expansion.bin', local_dir=fooocus_expansion_model_dir)
            os.rename(fooocus_expansion_model_dir / 'fooocus_expansion.bin', fooocus_expansion_model)
        except Exception:
            errors.report('### webui-fooocus-prompt-expansion: Failed to download model', exc_info=True)
            print(f'Download the model manually from "https://huggingface.co/lllyasviel/misc/tree/main/fooocus_expansion.bin" and place it in {fooocus_expansion_model_dir}.')


def safe_str(x):
    return re.sub(r' +', r' ', x).strip(",. \r\n")


def get_expansion_seed(seed_input, generation_seed):
    """Handle seed logic for expansion"""
    if seed_input == -1:
        # Generate random seed
        return random.randint(0, SEED_LIMIT_NUMPY - 1)
    elif seed_input == 0:
        # Use generation seed
        return generation_seed if generation_seed is not None else random.randint(0, SEED_LIMIT_NUMPY - 1)
    else:
        # Use provided seed
        return seed_input


def extract_added_tags(original, expanded):
    """Extract only the new tags that were added by expansion"""
    if not expanded or expanded == original:
        return ""
    
    # Find the last comma in original to use as anchor point
    last_comma_pos = original.rfind(',')
    
    if last_comma_pos != -1:
        # Get the text from last comma onwards in original
        anchor_text = original[last_comma_pos:]
        
        # Find this anchor in the expanded text (from the end)
        anchor_pos = expanded.rfind(anchor_text)
        
        if anchor_pos != -1:
            # Get everything after the anchor
            added_content = expanded[anchor_pos + len(anchor_text):].strip()
            if added_content.startswith(','):
                added_content = added_content[1:].strip()
            return added_content
    
    # Fallback: if expanded starts with original, extract the difference
    if expanded.startswith(original):
        added_content = expanded[len(original):].strip()
        if added_content.startswith(','):
            added_content = added_content[1:].strip()
        return added_content
    
    # Last resort: return entire expanded as "added"
    return expanded


class FooocusExpansion:
    def __init__(self):

        download_model()
        print(f'Loading models from {fooocus_expansion_model_dir}')
        self.tokenizer = AutoTokenizer.from_pretrained(fooocus_expansion_model_dir)

        positive_words = open(os.path.join(fooocus_expansion_model_dir, 'positive.txt'),
                              encoding='utf-8').read().splitlines()
        positive_words = ['Ġ' + x.lower() for x in positive_words if x != '']

        self.logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f'Fooocus V2 Expansion: Vocab with {len(debug_list)} words.')

        self.model = AutoModelForCausalLM.from_pretrained(fooocus_expansion_model_dir)
        self.model.eval()

        self.load_model_device = devices.get_optimal_device_name()
        use_fp16 = devices.dtype == torch.float16
        if use_fp16:
            self.model.half()

        self.model.to(self.load_model_device)  # Ensure the model is on the correct device

        print(f'Fooocus Expansion engine loaded for {self.load_model_device}, use_fp16 = {use_fp16}.')

    def unload_model(self):
        """Unload the model to free up memory."""
        del self.model
        torch.cuda.empty_cache()
        print('Model unloaded and memory cleared.')

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(self.load_model_device)

        bias = self.logits_bias.clone().to(self.load_model_device)  # Ensure bias is on the correct device
        bias[0, input_ids[0].to(self.load_model_device).long()] = neg_inf  # Ensure input_ids are on the correct device
        bias[0, 11] = 0

        return scores + bias.to(scores.device)  # Ensure bias is on the same device as scores

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt, seed):
        if not prompt:
            return ''

        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = safe_str(prompt) + ','
        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.load_model_device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.load_model_device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        features = self.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=LogitsProcessorList([self.logits_processor]))

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result


@lru_cache(maxsize=1024)
def create_positive(positive, seed):
    if not positive:
        return ''
    try:
        expansion = FooocusExpansion()
        positive = expansion(positive, seed=seed)
        expansion.unload_model()  # Unload the model after use
        return positive
    except Exception as e:
        print(f"❌ Fooocus Expansion Error in create_positive: {e}")
        return positive  # Return original prompt if error


def create_positive_with_weight(positive, seed, weight):
    """Create expansion with weight control"""
    if not positive:
        return ''
    
    try:
        # Get the expanded result from Fooocus
        expanded = create_positive(positive, seed)
        
        # If no expansion happened, return original
        if expanded == positive or not expanded:
            return positive
        
        # If weight is 1.0, return Fooocus result directly (no weight needed)
        if weight == 1.0:
            return expanded
        
        # For weights != 1.0, extract only the new tags and apply weight to them
        added_tags = extract_added_tags(positive, expanded)
        
        if added_tags:
            # Return: original + (new_tags:weight)
            return f"{positive}, ({added_tags}:{weight:.2f})"
        else:
            # No new tags found, return original
            return positive
        
    except Exception as e:
        print(f"❌ Fooocus Expansion Error in create_positive_with_weight: {e}")
        return positive  # Return original prompt if error


class FooocusPromptExpansion(scripts.Script):
    infotext_fields = []
    prompt_elm = None

    def __init__(self):
        super().__init__()
        self.on_after_component_elem_id = [
            ('txt2img_prompt', self.save_prompt_box),
            ('img2img_prompt', self.save_prompt_box),
        ]

    def title(self):
        return 'Fooocus Prompt Expansion'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Fooocus Expansion") as is_enabled:
            seed = gr.Number(value=-1, label="Expansion Seed", info="Seed for expansion: -1=random, 0=use generation seed, other=fixed seed")
            weight = gr.Number(value=1.0, label="Expansion Weight", info="Weight for expansion tags: 1.0=no weight, 0.75=(expansion:0.75)")
            if self.prompt_elm is not None:
                with gr.Row():
                    generate = gr.Button('Generate expansion prompts')
                    apply = gr.Button('Apply expansion to prompts')
                preview = gr.Textbox('', label="Expansion preview", interactive=False)

                for x in [preview, generate, apply]:
                    x.save_to_config = False

                generate.click(
                    fn=lambda prompt, seed_input, weight_val: self.safe_generate_preview(prompt, seed_input, weight_val),
                    inputs=[self.prompt_elm, seed, weight],
                    outputs=[preview],
                )
                apply.click(
                    fn=lambda prompt, seed_input, weight_val: self.safe_apply_expansion(prompt, seed_input, weight_val),
                    inputs=[self.prompt_elm, seed, weight],
                    outputs=[is_enabled, self.prompt_elm],
                )
        self.infotext_fields.append((is_enabled, lambda d: False))

        return [is_enabled, seed, weight]

    def safe_generate_preview(self, prompt, seed_input, weight_val):
        """Safely generate preview with error handling"""
        try:
            expansion_seed = get_expansion_seed(seed_input, None)
            return create_positive_with_weight(prompt, expansion_seed, weight_val)
        except Exception as e:
            print(f"❌ Fooocus Expansion Preview Error: {e}")
            return f"Error generating preview: {str(e)}"

    def safe_apply_expansion(self, prompt, seed_input, weight_val):
        """Safely apply expansion with error handling"""
        try:
            expansion_seed = get_expansion_seed(seed_input, None)
            expanded = create_positive_with_weight(prompt, expansion_seed, weight_val)
            return (False, expanded)
        except Exception as e:
            print(f"❌ Fooocus Expansion Apply Error: {e}")
            return (False, prompt)  # Return original prompt if error

    def process(self, p, is_enabled, seed_input, weight):
        if not is_enabled:
            return

        for i, prompt in enumerate(p.all_prompts):
            try:
                # Get the appropriate seed for this prompt
                if seed_input == -1:
                    # Generate random seed
                    expansion_seed = random.randint(0, SEED_LIMIT_NUMPY - 1)
                elif seed_input == 0:
                    # Use generation seed
                    generation_seed = getattr(p, 'seed', None)
                    if generation_seed is None:
                        generation_seed = random.randint(0, SEED_LIMIT_NUMPY - 1)
                    expansion_seed = generation_seed
                else:
                    # Use provided seed
                    expansion_seed = int(seed_input)
                
                # Get the expansion and apply weight
                expanded_prompt = create_positive_with_weight(prompt, expansion_seed, weight)
                
                # For logging, get the raw expansion and extract added tags
                raw_expansion = create_positive(prompt, expansion_seed)
                added_tags = extract_added_tags(prompt, raw_expansion)
                
                if added_tags:
                    if weight == 1.0:
                        print(f"✨ Fooocus Expansion added: {added_tags}")
                    else:
                        print(f"⚖️ Fooocus Expansion added (weight {weight}): {added_tags}")
                
                p.all_prompts[i] = expanded_prompt
                
            except Exception as e:
                print(f"❌ Fooocus Expansion Error processing prompt {i}: {e}")
                print(f"🔧 Keeping original prompt unchanged: {prompt}")
                # Keep original prompt unchanged if any error occurs
                continue

    def save_prompt_box(self, on_component):
        self.prompt_elm = on_component.component
