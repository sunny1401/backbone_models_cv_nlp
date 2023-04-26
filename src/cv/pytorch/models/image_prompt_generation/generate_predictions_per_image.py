import torch


def generate_captions(model, device, image_features, max_length=50, tokenizer=None):

    model.eval()
    # Adjust the image features' batch_size
    image_features = image_features.unsqueeze(0)
    # adding the initial inout tokens to the sentences
    input_tokens = torch.tensor(
        [tokenizer.bos_token_id]
    ).unsqueeze(0).to(device)

    # creating a dummy attn_mask which is needed by the model
    attn_mask = torch.ones_like(input_tokens)
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(image_features, input_tokens, attn_mask)

            # select the most probable token generated
            predicted_token = torch.argmax(logits[:, -1, :], dim=-1)
            # break off if the eos symbol is seen

            if predicted_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(predicted_token.item())
            # updating input tokens to take generated token into account
            input_tokens = torch.cat(
                (input_tokens, predicted_token.unsqueeze(0)), dim=1)
            # updating attn mask
            attn_mask = torch.ones_like(input_tokens)

    generated_text = tokenizer.decode(generated_tokens, skip_special_token=True)
    return generated_text
