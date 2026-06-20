
import os
import types
from typing import Any, List, Optional
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import spaces
# import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from diffusers.models.attention_processor import Attention

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16,
)

def encode_concepts(
    pipeline,
    concepts: list[str] = None,
    num_images_per_prompt: int = 1,
    clip_skip: Optional[int] = None,
    max_sequence_length: int = 256,
    device: Optional[torch.device] = None,
):
    device = device or pipeline._execution_device

    prompt = " ".join(concepts)
    prompt = [prompt] if isinstance(prompt, str) else prompt

    prompt_embed, _ = pipeline._get_clip_prompt_embeds(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
        clip_model_index=0,
    )
    prompt_2_embed, _ = pipeline._get_clip_prompt_embeds(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
        clip_model_index=1,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

    t5_prompt_embed = pipeline._get_t5_prompt_embeds(
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    prompt_embeds = prompt_embeds[:, : len(concepts) + 1]  # Leave room for padding token

    return prompt_embeds


def encode_concepts_prompts(concepts, prompt, pipe):
    concept_embeddings = encode_concepts(
        pipeline=pipe, 
        concepts=concepts,
        device=pipe.device,
        num_images_per_prompt=1
    )
    
    prompt_out_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return concept_embeddings, prompt_out_embeds


class CustomAttnProcessor:
    def __init__(self):
        self.store_out = {
            'concept_out': [],
            'image_out': [],
        }
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0 or newer.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        concept_hidden_states: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        img_q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            img_q = attn.norm_q(img_q)
        if attn.norm_k is not None:
            img_k = attn.norm_k(img_k)

        if encoder_hidden_states is not None:
            text_q = attn.add_q_proj(encoder_hidden_states)
            text_k = attn.add_k_proj(encoder_hidden_states)
            text_v = attn.add_v_proj(encoder_hidden_states)

            text_q = text_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            text_k = text_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            text_v = text_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                text_q = attn.norm_added_q(text_q)
            if attn.norm_added_k is not None:
                text_k = attn.norm_added_k(text_k)

        out_c = None
        if concept_hidden_states is not None:
            concept_q = attn.add_q_proj(concept_hidden_states)
            concept_k = attn.add_k_proj(concept_hidden_states)
            concept_v = attn.add_v_proj(concept_hidden_states)

            concept_q = concept_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            concept_k = concept_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            concept_v = concept_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                concept_q = attn.norm_added_q(concept_q)
            if attn.norm_added_k is not None:
                concept_k = attn.norm_added_k(concept_k)

            k_xc = torch.cat([img_k, concept_k], dim=2)
            v_xc = torch.cat([img_v, concept_v], dim=2)

            out_c = F.scaled_dot_product_attention(concept_q, k_xc, v_xc, dropout_p=0.0, is_causal=False)
            out_c = out_c.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            out_c = out_c.to(concept_q.dtype)

            self.store_out['concept_out'].append(out_c[1].detach().cpu().half())

        if encoder_hidden_states is not None:
            query = torch.cat([img_q, text_q], dim=2)
            key = torch.cat([img_k, text_k], dim=2)
            value = torch.cat([img_v, text_v], dim=2)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            self.store_out['image_out'].append(hidden_states[1].detach().cpu().half())

            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if concept_hidden_states is not None:
            return hidden_states, encoder_hidden_states, out_c
        elif encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def transformer_block_forward_patch(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_attention_kwargs = joint_attention_kwargs or {}
    concept_hidden_states = joint_attention_kwargs.get("concept_hidden_states", None)
    
    if self.use_dual_attention:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
            hidden_states, emb=temb
        )
    else:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    if self.context_pre_only:
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
    else:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

    if concept_hidden_states is not None:
        if self.context_pre_only:
            norm_concept_hidden_states = self.norm1_context(concept_hidden_states)
        else:
            norm_concept_hidden_states, _, _, _, _ = self.norm1_context(
                concept_hidden_states, emb=temb
            )
        joint_attention_kwargs["concept_hidden_states"] = norm_concept_hidden_states
        
    if concept_hidden_states is not None:
        attn_output, context_attn_output, concept_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )
    else:
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )

    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    if self.use_dual_attention:
        attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
        attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
        hidden_states = hidden_states + attn_output2

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    
    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = hidden_states + ff_output

    if self.context_pre_only:
        encoder_hidden_states = None
    else:
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        if concept_hidden_states is not None and not self.context_pre_only:
            concept_attn_output = c_gate_msa.unsqueeze(1) * concept_attn_output
            concept_hidden_states = concept_hidden_states + concept_attn_output
    
            norm_concept_hidden_states = self.norm2_context(concept_hidden_states)
            norm_concept_hidden_states = norm_concept_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            
            concept_ff_output = self.ff_context(norm_concept_hidden_states)
            concept_hidden_states = concept_hidden_states + c_gate_mlp.unsqueeze(1) * concept_ff_output
            
            joint_attention_kwargs["concept_hidden_states"] = concept_hidden_states
    
    return encoder_hidden_states, hidden_states


def apply_patched_fn(pipe, custom_processor):
    for i in range(23):
        block = pipe.transformer.transformer_blocks[i]
        block.attn.processor = custom_processor
        block.forward = types.MethodType(transformer_block_forward_patch, block)


def image_gen(pipe, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, concept_embeds, num_steps, img_size, custom_processor):
    concept_embeds = pipe.transformer.context_embedder(concept_embeds)

    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=7.0,
        height=img_size,
        width=img_size,
        joint_attention_kwargs={"concept_hidden_states": concept_embeds}
    )
    
    concept_vector = torch.stack(custom_processor.store_out['concept_out']).view(num_steps, 23, 1, -1, 1536)
    image_vector = torch.stack(custom_processor.store_out['image_out']).view(num_steps, 23, 1, -1, 1536)

    return concept_vector, image_vector, output.images[0]


def compute_heatmaps(
    img_info,
    concept_vectors: torch.Tensor,
    image_vectors: torch.Tensor,
    height: int,
    width: int,
    layer_indices: Optional[List[int]] = None,
    softmax: bool = True,
    average_over_timesteps: bool = False,
) -> torch.Tensor:
    if average_over_timesteps and len(concept_vectors.shape) == 5:
        concept_vectors = torch.mean(concept_vectors, dim=0)
        image_vectors = torch.mean(image_vectors, dim=0)

    num_layers = concept_vectors.shape[0]

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    concept_vectors = concept_vectors[:, :, 1:]
    concept_vectors = concept_vectors / (concept_vectors.norm(dim=-1, keepdim=True) + 1e-8)

    heatmaps = torch.einsum(
        "lbcd,lbpd->lbcp",
        concept_vectors.float(),
        image_vectors.float()
    )
    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=-2)

    heatmaps = heatmaps[layer_indices]
    heatmaps = heatmaps.mean(dim=0)
    heatmaps = heatmaps[0]

    h_tokens = height // (img_info['vae_scale_factor'] * img_info['patch_size'])
    w_tokens = width // (img_info['vae_scale_factor'] * img_info['patch_size'])

    heatmaps = heatmaps.view(-1, h_tokens, w_tokens)
    return heatmaps

custom_processor = CustomAttnProcessor()
apply_patched_fn(pipe, custom_processor)

@spaces.GPU(duration=60)
def inference_pipeline(prompt, concepts_string, steps, img_size):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
    
        custom_processor.store_out['concept_out'].clear()
        custom_processor.store_out['image_out'].clear()
        
        IMAGE_SIZE = int(img_size)
    
        concepts = [c.strip() for c in concepts_string.split(" ") if c.strip()]

        with torch.no_grad():
            concept_embeddings, (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = encode_concepts_prompts(concepts, prompt, pipe)
            # print(concept_embeddings.shape)
        
            concept_vector, image_vector, generated_image = image_gen(
                pipe=pipe,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                concept_embeds=concept_embeddings,
                num_steps=int(steps),
                img_size=IMAGE_SIZE,
                custom_processor=custom_processor
            )
        # print(concept_vector.shape, image_vector.shape)
    
        img_info = {
            "vae_scale_factor": pipe.vae_scale_factor,
            "patch_size": pipe.patch_size,
        }
        # del pipe
    
        heatmaps = compute_heatmaps(
            img_info,
            concept_vectors=concept_vector,
            image_vectors=image_vector,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            average_over_timesteps=True,
        )

    
        cols = (len(concepts) + 2) // 2
        fig, axes = plt.subplots(2, cols, figsize=(12, 6))
        axes_flat = axes.flatten()
    
        img = np.array(generated_image)
    
        for idx in range(len(concepts) + 1):
            if idx == 0:
                axes_flat[idx].imshow(img)
                axes_flat[idx].set_title("Original Image")
            else:
                hm = heatmaps[idx - 1].detach().cpu().numpy()
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    
                axes_flat[idx].imshow(hm, cmap="plasma")
                axes_flat[idx].set_title(concepts[idx - 1])
            axes_flat[idx].axis("off")
    
        for idx in range(len(concepts) + 1, len(axes_flat)):
            axes_flat[idx].axis("off")
    
        plt.tight_layout()
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise gr.Error(f"Pipeline Failed: {str(e)}")

    finally:
        import gc
        if 'custom_processor' in locals():
            custom_processor.store_out['concept_out'].clear()
            custom_processor.store_out['image_out'].clear()
        
        concept_vector = None
        image_vector = None
        output = None


ui = gr.Interface(
    fn=inference_pipeline,
    inputs=[
        gr.Textbox(label="Image Prompt", value="A dragon standing on a rock."),
        gr.Textbox(label="Concepts (Space-Separated)", value="dragon rock sky"),
        gr.Slider(minimum=1, maximum=30, value=28, step=1, label="SD3 Inference Steps"),
        gr.Slider(minimum=512, maximum=1024, value=768, step=64, label="Output Image Size"),
    ],
    outputs=gr.Plot(label="Concept Attention Maps"),
    title="ConceptAttention Visualization",
)

if __name__ == "__main__":
    ui.launch()