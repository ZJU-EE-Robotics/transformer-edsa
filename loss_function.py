import torch
from torch import nn


class TransformerLoss(nn.Module):
    def __init__(self, guide_attn=False, guide_scale=1.0, guide_sigma=0.4):
        super(TransformerLoss, self).__init__()
        self.guide_attn = guide_attn
        self.guide_scale = guide_scale
        self.sigma = guide_sigma

    def forward(self, model_outputs, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_output, mel_output_postnet, gate_output, _, _, attn = model_outputs[0]
        txt_flag, mel_flag = model_outputs[1], model_outputs[2]
        gate_output = gate_output.view(-1, 1)
        pos_weight = gate_output.new_full((1,), 6.0)

        # Custom defined weights for each loss
        mel_loss_l1 = nn.L1Loss()(mel_output, mel_target) + nn.L1Loss()(
            mel_output_postnet, mel_target
        )
        mel_loss_l2 = nn.MSELoss()(mel_output, mel_target) + nn.MSELoss()(
            mel_output_postnet, mel_target
        )
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(gate_output, gate_target)

        # Guide attention loss
        if self.guide_attn is True:
            guide_loss = self._get_guide_loss(attn, txt_flag, mel_flag)
        else:
            guide_loss = gate_loss.new_zeros(())

        return (
            mel_loss_l1 + mel_loss_l2 + gate_loss + guide_loss,
            (mel_loss_l1, mel_loss_l2, gate_loss, guide_loss),
        )

    def _get_guide_loss(self, attn, txt_flag, mel_flag):
        """
        Args:
            attn (List of Tensor): Attention weights (B, H, T_max_out, T_max_in).
            txt_flag (LongTensor): Batch of input lenghts flag (B, T_max_in).
            mel_flag (LongTensor): Batch of output lenghts flag  (B, T_max_out).

        Returns:
            Tensor: Guided attention loss value.
        """

        num = txt_flag.size(0)
        ilens = txt_flag.sum(dim=1)
        olens = mel_flag.sum(dim=1)
        padded_ilen = txt_flag.size(1)
        padded_olen = mel_flag.size(1)

        layer_loss = []
        for layer_id, layer_attn in enumerate(attn):

            # Guide the first 3 layers by default
            if layer_id < 6 and layer_id >= 3:
                # Guide the first head by default, (B, T_max_out, T_max_in)
                layer_attn = layer_attn[:, 0, :, :]
                guided_attn_masks = layer_attn.new_zeros((num, padded_olen, padded_ilen))

                for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
                    guided_attn_masks[
                        idx, :olen, :ilen
                    ] = self._make_guided_attention_mask(ilen, olen, self.sigma)
                losses = guided_attn_masks * layer_attn
                select_masks = (
                    mel_flag.unsqueeze(-1).bool() & txt_flag.unsqueeze(-2).bool()
                )
                losses = torch.mean(losses.masked_select(select_masks))
                layer_loss.append(losses)

        guide_loss = torch.mean(torch.stack(layer_loss))
        return self.guide_scale * guide_loss

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.

        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])

        """
        grid_x, grid_y = torch.meshgrid(
            torch.arange(olen, device=olen.device),
            torch.arange(ilen, device=ilen.device),
        )
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )
