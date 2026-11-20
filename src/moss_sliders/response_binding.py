"""MOSS binding extracted from the MIT-licensed shared core 0.13.0."""
import torch
from particle_sliders.backends import AdapterParameters, FrozenResponseAdapter

class MossResponseAdapter(FrozenResponseAdapter):
    family = "moss"
    time_coordinates = "official MOSS scheduler timestep = 1000*sigma"

    def __init__(self, transformer, slider, *, decoder=None, model_identity=None):
        self.transformer, self.slider = transformer, slider
        for block in getattr(transformer, "layers", ()):
            code = getattr(block.forward, "__code__", None)
            if (code is not None and "checkpoint" in (code.co_names + code.co_freevars) and
                    not getattr(block, "_adapter_scale_checkpoint", False)):
                raise ValueError("replace legacy checkpoint wrappers with checkpoint_with_adapter_scale")
        named = list(slider.named_parameters())
        if any(p.requires_grad for p in transformer.parameters()):
            raise ValueError("MOSS base transformer must be frozen")
        super().__init__(AdapterParameters(named, slider.at), decoder=decoder, model_identity=model_identity, teacher_convention="signed_difference", trajectory_direction="decreasing")

    def _velocity(self, condition, latent, timestep):
        if torch.any((timestep < 0) | (timestep > 1000)):
            raise ValueError("MOSS time must be the raw scheduler timestep in [0,1000]")
        embeds, mask = condition
        return self.transformer(latent, timestep, embeds, mask).sample.float()


def moss_pipeline_decoder(pipe, *, output_type='pil'):
    """Bind official MOSS FLUX.2 normalization/latent unpacking to the renderer."""
    if getattr(pipe, 'vae', None) is None:
        raise ValueError('a staged FLUX.2 VAE is required')

    def decode(latent):
        if not torch.isfinite(latent).all():
            raise FloatingPointError('nonfinite MOSS latent before VAE decode')
        vae = pipe.vae
        z = latent.to(device=vae.device, dtype=vae.dtype)
        bn = vae.bn
        z = (z * (bn.running_var + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
             + bn.running_mean.to(z).view(1, -1, 1, 1))
        batch, channels, height, width = z.shape
        z = (z.reshape(batch, channels // 4, 2, 2, height, width)
             .permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2))
        images = vae.decode(z, return_dict=False)[0]
        if not torch.isfinite(images).all():
            raise FloatingPointError('nonfinite MOSS VAE float output before clamp/postprocess')
        images = images.clamp(-1, 1)
        return pipe.image_processor.postprocess(images, output_type=output_type)
    return decode
