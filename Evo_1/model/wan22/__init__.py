__all__ = [
    "HuggingfaceTokenizer",
    "WanContinuousFlowMatchScheduler",
    "WanTextEncoder",
    "WanVideoVAE38",
]


def __getattr__(name):
    if name == "WanContinuousFlowMatchScheduler":
        from .schedulers import WanContinuousFlowMatchScheduler

        return WanContinuousFlowMatchScheduler
    if name in {"HuggingfaceTokenizer", "WanTextEncoder"}:
        from .wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

        return {"HuggingfaceTokenizer": HuggingfaceTokenizer, "WanTextEncoder": WanTextEncoder}[name]
    if name == "WanVideoVAE38":
        from .wan_video_vae import WanVideoVAE38

        return WanVideoVAE38
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
