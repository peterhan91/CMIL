import torch

def interpolate_pos_embed(model, checkpoint_model, old_size=(128, 224, 224)):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']  # shape: (1, N_old, embed_dim)
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        patch_size = model.patch_embed.patch_size

        old_D = old_size[0] // patch_size[0] # 128 // 4 = 32
        old_H = old_size[1] // patch_size[1] # 224 // 16 = 14
        old_W = old_size[2] // patch_size[2] # 224 // 16 = 14

        new_input_size = model.patch_embed.img_size
        new_D = new_input_size[0] // patch_size[0]
        new_H = new_input_size[1] // patch_size[1]
        new_W = new_input_size[2] // patch_size[2]

        if (old_D, old_H, old_W) != (new_D, new_H, new_W):
            print(f"Position interpolate from {old_D}x{old_H}x{old_W} to {new_D}x{new_H}x{new_W}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]  # shape: [1, D*H*W, embedding_size]
            pos_tokens = pos_tokens.reshape(1, old_D, old_H, old_W, embedding_size).permute(0, 4, 1, 2, 3)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_D, new_H, new_W), 
                mode='trilinear', align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1).reshape(1, new_D * new_H * new_W, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
