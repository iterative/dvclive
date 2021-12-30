import gorilla
import tensorflow as tf

from dvclive import Live

# pylint: disable=unused-argument, no-member


def patch_tensorboard(override: bool = True, **kwargs):
    live = Live(**kwargs)
    settings = gorilla.Settings(allow_hit=True, store_hit=True)

    original_scalar = gorilla.Patch(
        tf.summary, "original_scalar", tf.summary.scalar, settings=settings
    )
    gorilla.apply(original_scalar)

    def log_scalar(name, data, step=None, description=None):
        if step is not None:
            if step != live.get_step():
                live.set_step(step)
        live.log(name, data)
        if not override:
            tf.summary.original_scalar(name, data, step=None, description=None)

    original_image = gorilla.Patch(
        tf.summary, "original_image", tf.summary.image, settings=settings
    )
    gorilla.apply(original_image)

    def log_image(name, data, step=None, max_outputs=3, description=None):
        name += ".png"
        if step is not None:
            if step != live.get_step():
                live.set_step(step)
        if len(data) > 1:
            for n, image in enumerate(data):
                if n > max_outputs:
                    break
                live.log_image(f"sample-{n}-{name}", image)
        else:
            live.log_image(name, data[0])

        if not override:
            tf.summary.original_image(name, data, step=None, description=None)

    scalar_patch = gorilla.Patch(tf.summary, "scalar", log_scalar, settings)
    gorilla.apply(scalar_patch)

    image_patch = gorilla.Patch(tf.summary, "image", log_image, settings)
    gorilla.apply(image_patch)

    return original_scalar, original_image, scalar_patch, image_patch
