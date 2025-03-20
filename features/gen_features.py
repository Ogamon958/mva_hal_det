import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import LlamaForCausalLM, LlamaTokenizer
import ipywidgets as widgets
from IPython.display import display
import re
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import entropy
import torch.nn.functional as F

# Changed to default operation with fp16 (20241115)

# llama3_generate function
def llama3_generate(prompt, model, tokenizer, device, tem, k, p, seed_value=42, max_new_tokens=70):
    model.eval()
    with torch.no_grad():
        # Set seed value
        torch.manual_seed(seed_value)
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            temperature=tem,
            top_k=k,
            top_p=p,
            return_dict_in_generate=True,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        output_id_list = outputs.sequences[0].tolist()
        output_tokens = tokenizer.convert_ids_to_tokens(output_id_list)
        output_tensor = outputs.sequences[0]
        
        # Also output generated_text by removing input_ids from output_text (excluding the final eot_token)
        generated_text = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        del input_ids, attention_mask, outputs

    return output_text, output_tokens, output_tensor, generated_text

# Function to get attention (fp16 compatible)
def get_self_attention(input_tensor, model):
    model.eval()
    with torch.no_grad():
        attention_mask = torch.ones_like(input_tensor).to(input_tensor.device)

        # Enable processing with fp16
        with torch.cuda.amp.autocast():
            inputs = {'input_ids': input_tensor.unsqueeze(0), 'attention_mask': attention_mask.unsqueeze(0)}
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        # Free memory
        del attention_mask, inputs, outputs

    return attentions


def get_norm_all_layers(model, input_tensors, device_v, device_o, w_v_all, w_o_all, num_layers):
    model.eval()
    n_att_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    n_rep = n_att_heads // n_kv_heads

    embeddings = model.model.embed_tokens(input_tensors)
    embeddings = embeddings.to(device_v).to(dtype=torch.float16)

    individual_head_outputs_all_layers = []

    with torch.no_grad():
        for layer_num in range(num_layers):
            w_v = w_v_all[layer_num]
            w_o = w_o_all[layer_num]

            a, b = w_v.shape
            w_v_aug = w_v[:, None, :].expand(a, n_rep, b).reshape(a * n_rep, b)

            head_dim = model.config.hidden_size // n_att_heads
            individual_head_outputs = []

            for h in range(n_att_heads):
                v_weight_head = w_v_aug[h * head_dim:(h + 1) * head_dim, :]
                o_weight_head = w_o[h * head_dim:(h + 1) * head_dim, :]

                # Minimize data transfer between devices
                v_head = torch.matmul(embeddings, v_weight_head.transpose(0, 1))
                head_output = torch.matmul(v_head.to(device_o), o_weight_head).to(device_v)

                individual_head_outputs.append(head_output)

                # Delete unnecessary tensors to free memory
                del v_weight_head, o_weight_head, v_head, head_output
                torch.cuda.empty_cache()

            # Stack the outputs of individual heads for each layer
            individual_head_outputs = torch.stack(individual_head_outputs)
            individual_head_outputs_all_layers.append(individual_head_outputs)

            # Free memory
            del w_v, w_o, w_v_aug, individual_head_outputs
            torch.cuda.empty_cache()

    # Finally stack the outputs for each layer
    individual_head_outputs_all_layers = torch.stack(individual_head_outputs_all_layers)
    del embeddings  # Free other memory as necessary
    torch.cuda.empty_cache()

    return individual_head_outputs_all_layers  # f(x) for each head



def get_f_lx_all_layers(individual_head_outputs_all_layers):
    # Calculate the norm magnitude for all layers and arrange into a 2D array (calculate ||f(x)||)
    # individual_head_outputs_all_layers has shape (num_layers, num_heads, num_tokens, hidden_dim) -> torch.Size([32, 32, 83, 4096])

    all_token_norms = []

    with torch.no_grad():
        for layer_outputs in individual_head_outputs_all_layers:
            token_norms = []
            for head_output in layer_outputs:
                # Calculate the norm and create a list of norms for each token
                head_norms = torch.norm(head_output, dim=1).cpu()  # Calculate the norm of a 4096-dimensional vector
                #print(head_norms)
                token_norms.append(head_norms)
            
            token_norms = torch.stack(token_norms)
            
            all_token_norms.append(token_norms)
    
    all_token_norms = torch.stack(all_token_norms)

    
    return all_token_norms



def get_alpha_fx_all_layers_optimized(token_norms_all_layers, attentions, device_v, device_o=None, mode="single"):
    """
    Calculate alpha_f_all_layers by element-wise multiplying token_norms_all_layers and attentions.
    Switch operation according to mode:
        - "single": Process all data on a single GPU.
        - "half_split": Split data in half and process sequentially on one GPU.
        - "multi_gpu": Split data in half and process in parallel on multiple GPUs.

    Args:
        token_norms_all_layers (torch.Tensor): [num_layers, num_heads, num_tokens]
        attentions (torch.Tensor): [num_layers, num_heads, num_tokens]
        device_v (str): Main GPU device (e.g., "cuda:0")
        device_o (str, optional): Secondary GPU device (e.g., "cuda:1")
        mode (str): Operation mode ("single", "half_split", "multi_gpu")

    Returns:
        torch.Tensor: alpha_f_all_layers
    """
    # Convert tensors to fp16 and adjust dimensions
    token_norms_all_layers = token_norms_all_layers.unsqueeze(3).half()  # [num_layers, num_heads, num_tokens, 1]
    attentions = attentions.half().squeeze(1)  # [num_layers, num_heads, num_tokens, 1]

    num_tokens = attentions.shape[2]
    half_tokens = num_tokens // 2

    # Process without calculating gradients
    with torch.no_grad():
        if mode == "single":
            # Process all data on one GPU
            token_norms_all_layers = token_norms_all_layers.to(device_v)
            attentions = attentions.to(device_v)
            alpha_f_all_layers = torch.mul(token_norms_all_layers, attentions).to(device_o)

        elif mode == "half_split":
            # Process sequentially by splitting into two halves (on one GPU)
            token_norms_all_layers = token_norms_all_layers.to(device_v)
            attentions = attentions.to(device_v)

            # Process first half
            alpha_f_all_layers_first_half = torch.mul(
                token_norms_all_layers[:, :, :half_tokens, :],
                attentions[:, :, :half_tokens, :]
            ).to("cpu")  # Move to CPU
            torch.cuda.empty_cache()  # Free GPU memory

            # Process second half
            alpha_f_all_layers_second_half = torch.mul(
                token_norms_all_layers[:, :, half_tokens:, :],
                attentions[:, :, half_tokens:, :]
            ).to("cpu")  # Move to CPU
            torch.cuda.empty_cache()

            # Concatenate on CPU and move to secondary device
            alpha_f_all_layers = torch.cat(
                (alpha_f_all_layers_first_half, alpha_f_all_layers_second_half), dim=2
            ).to(device_o)

        elif mode == "multi_gpu":
            if device_o is None:
                raise ValueError("Mode 'multi_gpu' requires a secondary device specified as 'device_o'.")

            # Process first half on device_v
            token_norms_first_half = token_norms_all_layers[:, :, :half_tokens, :].to(device_v)
            attentions_first_half = attentions[:, :, :half_tokens, :].to(device_v)
            alpha_f_all_layers_first_half = torch.mul(token_norms_first_half, attentions_first_half)
            del token_norms_first_half, attentions_first_half
            torch.cuda.synchronize(device_v)
            torch.cuda.empty_cache()

            # Process second half on device_o
            token_norms_second_half = token_norms_all_layers[:, :, half_tokens:, :].to(device_o)
            attentions_second_half = attentions[:, :, half_tokens:, :].to(device_o)
            alpha_f_all_layers_second_half = torch.mul(token_norms_second_half, attentions_second_half)
            del token_norms_second_half, attentions_second_half
            torch.cuda.synchronize(device_o)
            torch.cuda.empty_cache()

            # Move first half to secondary device (device_o) and concatenate
            alpha_f_all_layers_first_half = alpha_f_all_layers_first_half.to(device_o)
            alpha_f_all_layers = torch.cat(
                (alpha_f_all_layers_first_half, alpha_f_all_layers_second_half), dim=2
            )

            # Delete unnecessary data
            del alpha_f_all_layers_first_half, alpha_f_all_layers_second_half
            torch.cuda.synchronize()

        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes: 'single', 'half_split', 'multi_gpu'.")

    return alpha_f_all_layers



# Function to remove 'Ġ'
def remove_gerund_token(input_str):
    # Remove 'Ġ' that appears when the token's first character is a whitespace
    return input_str.replace("Ġ", "")


# Attention visualization function
def visualize_attention(attentions, alpha_f_all_layers, output_tokens, token_norms_all_layers, logscaled=False, layer_idx=0, head_idx=0):
    # attentions: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attention_map = attentions[layer_idx][0][head_idx].detach().cpu().numpy()
    alpha_f_map = alpha_f_all_layers[layer_idx][head_idx].detach().cpu().numpy()
    token_norms = token_norms_all_layers[layer_idx][head_idx].detach().cpu().numpy()
    
    # Remove 'Ġ' from token names
    cleaned_tokens = [remove_gerund_token(token) for token in output_tokens]

    # Scale norm values by 100
    token_norms_scaled = token_norms * 100
        
    # Convert zero values to 1e-6
    attention_map[attention_map == 0] = 1e-6
    alpha_f_map[alpha_f_map == 0] = 1e-6
    token_norms_scaled[token_norms_scaled == 0] = 1e-6
    
    # Prepare plot - use Gridspec to arrange axes0 and axes1 side by side, with axes2 below
    fig = plt.figure(figsize=(20, 10))  # Set overall figure size here
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 0.3])  # Two rows on top, one row (0.3) at bottom

    # Adjust vertical spacing (hspace) between grids
    gs.update(hspace=0.8)

    if logscaled:
        # Heatmap for Attention Map (top left) - apply logarithmic scale
        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(attention_map, xticklabels=cleaned_tokens, yticklabels=cleaned_tokens, cmap="viridis",
                    norm=LogNorm(vmin=attention_map.min() + 1e-6, vmax=attention_map.max()), ax=ax0)
        ax0.set_title(f"Attention Map - Layer {layer_idx + 1}, Head {head_idx + 1}")
        ax0.set_xlabel("Key Tokens")
        ax0.set_ylabel("Query Tokens")

        # Heatmap for Alpha_f Map (top right) - apply logarithmic scale
        ax1 = fig.add_subplot(gs[0, 1])
        sns.heatmap(alpha_f_map, xticklabels=cleaned_tokens, yticklabels=cleaned_tokens, cmap="viridis",
                    norm=LogNorm(vmin=alpha_f_map.min() + 1e-6, vmax=alpha_f_map.max()), ax=ax1)
        ax1.set_title(f"Alpha F Map - Layer {layer_idx + 1}, Head {head_idx + 1}")
        ax1.set_xlabel("Key Tokens")
        ax1.set_ylabel("Query Tokens")

        # Line graph of Token Norms (bottom) - color changes on logarithmic scale
        ax2 = fig.add_subplot(gs[1, :])  # Combine two columns to span full width
        combined_data = np.expand_dims(token_norms_scaled, axis=0)  # Increase dimension to make it one row
        im = ax2.imshow(combined_data, aspect='auto', cmap="autumn_r", norm=LogNorm(vmin=token_norms_scaled.min() + 1e-6, vmax=token_norms_scaled.max()))

        # Set axes and labels
        ax2.set_xticks(np.arange(len(cleaned_tokens)))
        ax2.set_xticklabels(cleaned_tokens, rotation=90)
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Norms (f)'])

        # Display values (scaled by 100) with 2 decimal places
        for i in range(len(token_norms_scaled)):
            value = f"{token_norms_scaled[i]:.2f}"
            ax2.text(i, 0, value, ha='center', va='center', color='black')

        # Add color bar (adjusted for 100x scaled range)
        cbar = plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Norm Value (scaled by 100) - Log Scale')
    
    else:
        # Heatmap for Attention Map (top left)
        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(attention_map, xticklabels=cleaned_tokens, yticklabels=cleaned_tokens, cmap="viridis", ax=ax0)
        ax0.set_title(f"Attention Map - Layer {layer_idx + 1}, Head {head_idx + 1}")
        ax0.set_xlabel("Key Tokens")
        ax0.set_ylabel("Query Tokens")

        # Heatmap for Alpha_f Map (top right)
        ax1 = fig.add_subplot(gs[0, 1])
        sns.heatmap(alpha_f_map, xticklabels=cleaned_tokens, yticklabels=cleaned_tokens, cmap="viridis", ax=ax1)
        ax1.set_title(f"Alpha F Map - Layer {layer_idx + 1}, Head {head_idx + 1}")
        ax1.set_xlabel("Key Tokens")
        ax1.set_ylabel("Query Tokens")

        # Line graph of Token Norms (bottom)
        ax2 = fig.add_subplot(gs[1, :])  # Combine two columns to span full width
        combined_data = np.expand_dims(token_norms_scaled, axis=0)  # Increase dimension to make it one row
        im = ax2.imshow(combined_data, aspect='auto', cmap="autumn_r")  # Use reversed 'autumn' colormap

        # Set axes and labels
        ax2.set_xticks(np.arange(len(cleaned_tokens)))
        ax2.set_xticklabels(cleaned_tokens, rotation=90)
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Norms (f)'])

        # Display values (scaled by 100) with 2 decimal places
        for i in range(len(token_norms_scaled)):
            value = f"{token_norms_scaled[i]:.2f}"
            ax2.text(i, 0, value, ha='center', va='center', color='black')

        # Add color bar (corresponding to the 100x scaled range)
        cbar = plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Norm Value (scaled by 100)')

    #fig.tight_layout()
    plt.show()


# Widget to interactively select layer and head.
# Visualize attention and alpha_f.
def interactive_attention_visualization(prompt, model, tokenizer, device, device_v, device_o, tem, k, p, max_new_tokens, logscaled=False):
    # Generate text
    output_text, output_tokens, output_tensor, generated_text = llama3_generate(
        prompt, model, tokenizer, device, tem, k, p, max_new_tokens
    )

    # Get data type and number of layers
    dtype = torch.float16
    num_layers = model.config.num_hidden_layers

    # Transfer v_proj and o_proj weights to device and stack the weights for each layer into a tensor
    with torch.no_grad():
        w_v_all = torch.stack([model.model.layers[i].self_attn.v_proj.weight.to(device_v, dtype=dtype, non_blocking=True) 
                            for i in range(num_layers)])
        w_o_all = torch.stack([model.model.layers[i].self_attn.o_proj.weight.to(device_o, dtype=dtype, non_blocking=True) 
                            for i in range(num_layers)])

        # Obtain attention
        attentions = get_self_attention(output_tensor, model)
        
        # Stack the tuple (shape: tuple of 32 layers), each tuple transformed to shape torch.Size([32, 1, 32, 83, 83])
        attentions = torch.stack(attentions)
        num_tokens = len(output_tokens)


    # Move output_tensor to device and process
    individual_head_outputs_all_layers = get_norm_all_layers(model, output_tensor.to(device_v), device_v, device_o, w_v_all, w_o_all, num_layers)
    token_norms_all_layers = get_f_lx_all_layers(individual_head_outputs_all_layers)
    alpha_f_all_layers = get_alpha_fx_all_layers_optimized(token_norms_all_layers, attentions, device_v, device_o, mode="single")
    
    # Create widget (for layer and head selection)
    layer_slider = widgets.IntSlider(value=0, min=0, max=len(attentions)-1, step=1, description='Layer')
    head_slider = widgets.IntSlider(value=0, min=0, max=attentions[0].size(1)-1, step=1, description='Head')

    # Function that is called when the widget is interacted with
    def update_visualization(layer_idx, head_idx):
        visualize_attention(attentions, alpha_f_all_layers, output_tokens, token_norms_all_layers, logscaled, layer_idx=layer_idx, head_idx=head_idx)

    # Create a tab
    ui = widgets.VBox([layer_slider, head_slider])
    out = widgets.interactive_output(update_visualization, {'layer_idx': layer_slider, 'head_idx': head_slider})

    # Display
    display(ui, out)
    
    print("prompt:\n", prompt, "\n")
    print("generated_text:\n", generated_text, "\n")


""" Example execution
# Example prompt
prompt = "This is an example prompt to generate text and visualize attention."

# Execute interactive visualization
interactive_attention_visualization(prompt, model, tokenizer, device, device_v, device_o)
"""


# Divide the generated text into prompt, input text, and generated text parts
def find_prompt_and_input_and_output_occurrences(lst):
    # Up to the second <|end_header_id|> is the prompt part
    # Up to the third <|end_header_id|> is the input text part
    #target_string = '<|end_header_id|>'
    
    # For llama3
    target_string_llama = '<|start_header_id|>'

    # For qwen
    target_string_qwen = '<|im_start|>'
    
    
    # List to record occurrence positions
    split_token_positions = []

    # Check the list one by one
    for i, s in enumerate(lst):
        # If it matches the specified string
        if s == (target_string_llama or target_string_qwen):
            split_token_positions.append(i)
            # Stop after finding the third occurrence
            if len(split_token_positions) == 3:
                break
    
    return split_token_positions

# Divide the generated text into input text part and generated text part.
# <|end_header_id|> appears in the prompt part and question part; following that is a Ċ token representing a newline.
# Return the position of the Ċ token in the generated text; everything up to that is treated as input text, and everything after as generated text.
def find_split_position(lst):
    
    # Defined target tokens
    # For llama3
    end_token_llama = '<|end_header_id|>'
    
    # For qwen
    end_token_qwen = '<|im_start|>'
    
    # List to record token positions
    end_token_positions = []

    # Check the list one by one to find the position of the end token
    for i, s in enumerate(lst):
        if s == (end_token_llama or end_token_qwen):
            end_token_positions.append(i)
            if s == end_token_llama:
                token_type = 'llama'
            elif s == end_token_qwen:
                token_type = 'qwen'
            # Stop after finding the third end token
            if len(end_token_positions) == 3:
                break

    if len(end_token_positions) < 3:
        raise ValueError("3rd end token not found.")

    # Find the first `Ċ` token after the third end token
    if token_type == 'llama':
        for i in range(end_token_positions[2] + 1, len(lst)):
            if lst[i] == 'Ċ':
                return i  # Return the position of the `Ċ` token
    elif token_type == 'qwen':
        for i in range(end_token_positions[2] + 2, len(lst)):
            if lst[i] == 'Ċ':
                return i

    # Raise an error if the `Ċ` token is not found
    raise ValueError("Ċ token not found after the 3rd end token.")


# Function to calculate entropy for a 1D tensor (tensor-compatible)
def calculate_entropy_batch(d2_tensor):
    # Convert tensor to float32
    d2_tensor = d2_tensor.float()

    # Clip small values to avoid issues with zero elements
    prob_dist = d2_tensor / torch.sum(d2_tensor, dim=-1, keepdim=True)
    prob_dist = prob_dist.clamp(min=1e-10)

    # Calculate entropy in batch
    return -torch.sum(prob_dist * torch.log2(prob_dist), dim=-1)

# Calculate entropy
def cal_entropy_score(attentions, output_tokens, split_token_positions, alpfa_f_flag=False, device='cuda'):
    # Transfer attentions to the specified device (GPU)
    attentions = [att.to(device) for att in attentions]

    # Align tensor dimensions for batch processing
    if alpfa_f_flag:
        head_dim = attentions[0].shape[0]
    else:
        head_dim = attentions[0].shape[1]

    step_count = len(output_tokens) - split_token_positions[2]

    # Create a tensor with dimensions (num_layers, head, step_count, 8) at once
    entropy_tensor = torch.zeros((len(attentions), head_dim, step_count, 8), device=device, dtype=torch.float32)

    for layer_idx, attention_layer in enumerate(attentions):
        for head_idx in range(head_dim):
            # Select attention_map based on alpfa_f_flag
            attention_map = attention_layer[head_idx] if alpfa_f_flag else attention_layer[0][head_idx]

            # Calculate entropy for all steps in the generated part at once
            attention_steps = attention_map[split_token_positions[2]:]
            entropy_tensor[layer_idx][head_idx][:, 0] = attention_steps[:, 0]
            entropy_tensor[layer_idx][head_idx][:, 1] = torch.mean(attention_steps[:, split_token_positions[0]:split_token_positions[1]], dim=1)
            entropy_tensor[layer_idx][head_idx][:, 2] = torch.mean(attention_steps[:, split_token_positions[1]:split_token_positions[2]], dim=1)
            entropy_tensor[layer_idx][head_idx][:, 3] = torch.mean(attention_steps[:, split_token_positions[2]:], dim=1)
            entropy_tensor[layer_idx][head_idx][:, 4] = calculate_entropy_batch(attention_steps[:, split_token_positions[0]:split_token_positions[1]])
            entropy_tensor[layer_idx][head_idx][:, 5] = calculate_entropy_batch(attention_steps[:, split_token_positions[1]:split_token_positions[2]])
            entropy_tensor[layer_idx][head_idx][:, 6] = calculate_entropy_batch(attention_steps[:, split_token_positions[2]:])
            entropy_tensor[layer_idx][head_idx][:, 7] = calculate_entropy_batch(attention_steps[:, :])

    # Transfer to CPU at the end
    entropy_tensor = entropy_tensor.cpu()
             
    return entropy_tensor



import torch

def get_features(attentions, output_tokens, alpfa_f_flag=False, device='cuda'):
    # Transfer attentions to the specified device (GPU)
    attentions = [att.to(device) for att in attentions]
    
    # Create a tensor with line numbers starting from 1
    row_indices = torch.arange(1, len(output_tokens) + 1, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Align tensor dimensions for batch processing
    head_dim = attentions[0].shape[0] if alpfa_f_flag else attentions[0].shape[1]

    # Create tensors for (num_layers, head, steps) at once (in float16)
    key_avg = torch.zeros((len(attentions), head_dim, len(output_tokens)), device=device, dtype=torch.float16)
    query_entropy = torch.zeros((len(attentions), head_dim, len(output_tokens)), device=device, dtype=torch.float16)
    key_entropy = torch.zeros((len(attentions), head_dim, len(output_tokens)), device=device, dtype=torch.float16)
    lookback_ratio = torch.zeros((len(attentions), head_dim, len(output_tokens)), device=device, dtype=torch.float16)

    def normalized_entropy_with_log(input_tensor, dim=-1, eps=1e-9):
        """
        Function that calculates entropy ignoring 0 elements, and normalizes by dividing by the log of the number of non-zero elements.
        Args:
            input_tensor (torch.Tensor): Input tensor (float16)
            dim (int): Dimension over which to calculate entropy
            eps (float): A small value to prevent division by zero
        Returns:
            torch.Tensor: Entropy normalized by the log of non-zero count (float16)
        """
        # Create a mask (to ignore 0 elements)
        mask = (input_tensor != 0).float()
        
        # Apply mask to input tensor and cast to float32
        masked_input = (input_tensor * mask).float()

        # Convert to probability distribution
        sum_masked_input = torch.sum(masked_input, dim=dim, keepdim=True) + eps  # float32
        probabilities = masked_input / sum_masked_input  # float32

        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps) * mask.float(), dim=dim)  # float32

        # Calculate the number of non-zero elements
        non_zero_count = torch.sum(mask, dim=dim).clamp(min=1)  # float32

        # Calculate log(number of non-zero elements)
        log_non_zero_count = torch.log(non_zero_count + eps)  # float32

        # Normalize entropy only when non_zero_count > 1, otherwise set to 0
        normalized_entropy = torch.where(
            non_zero_count > 1,
            entropy / log_non_zero_count,
            torch.zeros_like(entropy)
        )  # float32

        # Cast back to float16
        normalized_entropy = normalized_entropy.to(torch.float16)

        return normalized_entropy

    def normalize_with_zeros(input_tensor, dim=-1, eps=1e-9):
        """
        Function that ignores 0 elements when computing normalization, and maintains zeros in the normalized tensor.
        Args:
            input_tensor (torch.Tensor): Input tensor (float16)
            dim (int): Dimension over which to apply normalization
            eps (float): A small value to prevent division by zero
        Returns:
            torch.Tensor: Normalized tensor (float16)
        """
        # Create a mask (to ignore 0 elements)
        mask = (input_tensor != 0).float()
        
        # Apply mask to input tensor and cast to float32
        masked_input = (input_tensor * mask).float()
        
        # Calculate sum of non-zero elements
        sum_masked_input = torch.sum(masked_input, dim=dim, keepdim=True) + eps  # float32

        # Normalize
        normalized_output = masked_input / sum_masked_input  # float32

        # Apply mask to maintain zeros and cast back to float16
        normalized_output = normalized_output * mask  # float32
        normalized_output = normalized_output.to(torch.float16)

        return normalized_output
    
    def calculate_lookback_ratio(attentions, output_tokens, lookback_ratio, layer_idx, head_idx):
        
        split_position = find_split_position(output_tokens)
        context_length = split_position + 1
        new_token_length = len(output_tokens) - context_length
        
        """
        print(lookback_ratio.shape)
        print(context_length, len(output_tokens))
        print(new_token_length)
        print(output_tokens[split_position])
        """
        
        for i in range(new_token_length):  # iterating over the new tokens length
            attn_on_context = attentions[:context_length, i].mean(-1)
            attn_on_new_tokens = attentions[context_length:(context_length + i + 1), i].mean(-1)
            lookback_ratio[layer_idx, head_idx, (context_length + i)] = attn_on_context / (attn_on_context + attn_on_new_tokens)
            

    # Extract attention data
    for layer_idx, attention_layer in enumerate(attentions):
        for head_idx in range(head_dim):
            # Get attention_map for each head
            attention_map = attention_layer[head_idx] if alpfa_f_flag else attention_layer[0][head_idx]
            
            calculate_lookback_ratio(attention_map, output_tokens, lookback_ratio, layer_idx, head_idx)
            
            # Keep original attention_map (for mean calculation)
            original_attention_map = attention_map.clone()

            # Normalize along rows
            attention_map = normalize_with_zeros(attention_map, dim=1)
            
            # Calculate row-wise entropy (float16)
            query_entropy[layer_idx][head_idx][:] = normalized_entropy_with_log(attention_map, dim=1)
            
            # Multiply according to row number (calculated in float32)
            attention_map_float32 = (original_attention_map * row_indices).float()
            
            # Calculate mean across columns (using original, non-normalized values)
            mean_attention = torch.mean(attention_map_float32, dim=0)
            key_avg[layer_idx][head_idx][:] = mean_attention.to(torch.float16)
            
            # Normalize along columns (using original_attention_map again)
            attention_map = normalize_with_zeros(original_attention_map, dim=0)
            
            # Calculate column-wise entropy (float16)
            key_entropy[layer_idx][head_idx][:] = normalized_entropy_with_log(attention_map, dim=0)
            
    # Transfer to CPU and replace NaN with 0
    key_avg_cpu = key_avg.cpu().nan_to_num(nan=0.0)
    query_entropy_cpu = query_entropy.cpu().nan_to_num(nan=0.0)
    key_entropy_cpu = key_entropy.cpu().nan_to_num(nan=0.0)
    lookback_ratio_cpu = lookback_ratio.cpu().nan_to_num(nan=0.0)
    
    return key_avg_cpu, query_entropy_cpu, key_entropy_cpu, lookback_ratio_cpu




import torch

def get_features_with_generate(prompt, model, tokenizer, device, device_v, device_o, tem, k, p, seed_value, max_new_tokens):
    """
    Function to generate text and compute various features during generation.

    Parameters:
    - prompt (str): Prompt input to the model.
    - model: Language model used.
    - tokenizer: Tokenizer corresponding to the model.
    - device (torch.device): Primary computation device for the model.
    - device_v (torch.device): Device used for v_proj computation.
    - device_o (torch.device): Device used for o_proj computation.
    - tem (float): Temperature parameter.
    - k (int): Top-k filtering parameter.
    - p (float): Top-p filtering parameter.
    - seed_value (int): Seed value.
    - max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
    - tuple: Tuple of generated text and computed features.
      (
          output_text,               # Generated text
          generated_text,            # Detailed generated text
          output_tokens,             # Generated tokens
          raw_key_avg,               # Raw attention values
          raw_query_entropy,         # Raw query entropy
          raw_key_entropy,           # Raw key entropy
          raw_lookback_lens,         # Raw lookback lengths
          norm_key_avg,              # Normalized attention values
          norm_query_entropy,        # Normalized query entropy
          norm_key_entropy,          # Normalized key entropy
          norm_lookback_lens         # Normalized lookback lengths
      )
    """

    # Generate text
    output_text, output_tokens, output_tensor, generated_text = llama3_generate(
        prompt, model, tokenizer, device, tem, k, p, seed_value, max_new_tokens
    )

    # Get data type and number of layers
    dtype = torch.float16
    num_layers = model.config.num_hidden_layers

    # Transfer v_proj and o_proj weights to each device and stack the weights for each layer into a tensor
    with torch.no_grad():
        w_v_all = torch.stack([
            model.model.layers[i].self_attn.v_proj.weight.to(device_v, dtype=dtype, non_blocking=True) 
            for i in range(num_layers)
        ])
        w_o_all = torch.stack([
            model.model.layers[i].self_attn.o_proj.weight.to(device_o, dtype=dtype, non_blocking=True) 
            for i in range(num_layers)
        ])

        # Obtain attention
        attentions = get_self_attention(output_tensor, model)
        
        # Stack attentions (shape: [num_layers, batch_size, num_heads, num_tokens, num_tokens])
        attentions = torch.stack(attentions)
        num_tokens = len(output_tokens)

    # Move output_tensor to the primary device and process
    individual_head_outputs_all_layers = get_norm_all_layers(
        model, output_tensor.to(device), device_v, device_o, w_v_all, w_o_all, num_layers
    )
    token_norms_all_layers = get_f_lx_all_layers(individual_head_outputs_all_layers)
    
    # Compute alpha_f_all_layers (multi_gpu mode)
    alpha_f_all_layers = get_alpha_fx_all_layers_optimized(
        token_norms_all_layers, attentions, device_v, device_o, mode="multi_gpu"
    )
    
    # Obtain features
    raw_key_avg, raw_query_entropy, raw_key_entropy, raw_lookback_lens = get_features(
        attentions, output_tokens, False, device_o
    )
    norm_key_avg, norm_query_entropy, norm_key_entropy, norm_lookback_lens = get_features(
        alpha_f_all_layers, output_tokens, True, device_o
    )
    
    # Reset GPU memory
    reset_gpu_memory()
    
    return (
        output_text,           # Generated text
        generated_text,        # Detailed generated text
        output_tokens,         # Generated tokens
        raw_key_avg,           # Raw attention values
        raw_query_entropy,     # Raw query entropy
        raw_key_entropy,       # Raw key entropy
        raw_lookback_lens,     # Raw lookback lengths
        norm_key_avg,          # Normalized attention values
        norm_query_entropy,    # Normalized query entropy
        norm_key_entropy,      # Normalized key entropy
        norm_lookback_lens     # Normalized lookback lengths
    )


# Function to release GPU memory after usage
def reset_gpu_memory():
    torch.cuda.empty_cache()  # Free cached memory
    torch.cuda.ipc_collect()  # Release memory of unused tensors


# Function to obtain features (without generation)
def get_features_no_generate(prompt, model, tokenizer, device, device_v, device_o):
    # Convert to tensor, etc.
    output_tensor = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    output_id_list = output_tensor.tolist()
    output_tokens = tokenizer.convert_ids_to_tokens(output_id_list)
    
    output_tensor = output_tensor.to(device)
    
    # Get data type and number of layers
    dtype = torch.float16
    num_layers = model.config.num_hidden_layers

    # Transfer v_proj and o_proj weights to device and stack the weights for each layer into a tensor
    with torch.no_grad():
        w_v_all = torch.stack([model.model.layers[i].self_attn.v_proj.weight.to(device_v, dtype=dtype, non_blocking=True) 
                            for i in range(num_layers)])
        w_o_all = torch.stack([model.model.layers[i].self_attn.o_proj.weight.to(device_o, dtype=dtype, non_blocking=True) 
                            for i in range(num_layers)])

        # Obtain attention
        attentions = get_self_attention(output_tensor, model)
        
        # Stack the tuple (shape: tuple of 32 layers), each tuple transformed to shape torch.Size([32, 1, 32, 83, 83])
        attentions = torch.stack(attentions)
        num_tokens = len(output_tokens)

    # Move output_tensor to device and process
    individual_head_outputs_all_layers = get_norm_all_layers(model, output_tensor, device_v, device_o, w_v_all, w_o_all, num_layers)
    token_norms_all_layers = get_f_lx_all_layers(individual_head_outputs_all_layers)
    
    
    alpha_f_all_layers = get_alpha_fx_all_layers_optimized(token_norms_all_layers, attentions, device_v, device_o, mode="multi_gpu")
    
    
    # Get token positions for the prompt, input, and output parts
    # token_positions = find_prompt_and_input_and_output_occurrences(output_tokens)
    
    
    raw_key_avg, raw_query_entropy, raw_key_entropy, raw_lookback_lens = get_features(attentions, output_tokens, False, device_o)
    norm_key_avg, norm_query_entropy, norm_key_entropy, norm_lookback_lens = get_features(alpha_f_all_layers, output_tokens, True, device_o)
    
    # Call after computation completes
    reset_gpu_memory()
    
    return output_tokens, raw_key_avg, raw_query_entropy, raw_key_entropy, raw_lookback_lens, norm_key_avg, norm_query_entropy, norm_key_entropy, norm_lookback_lens
