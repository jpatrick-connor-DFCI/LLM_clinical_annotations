"""Provider registry. Adapters are imported lazily so selecting one provider
never pulls in the other's SDK (an Azure run never imports google-genai, and
vice versa)."""

_PROVIDER_MODULES = {
    "dfci_gpt": "providers.dfci_gpt",
    "vertex_ai": "providers.vertex_ai",
}


def get_provider(name):
    """Return the adapter module for `name`, importing it on first use."""
    if name not in _PROVIDER_MODULES:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {sorted(_PROVIDER_MODULES)}"
        )
    import importlib

    return importlib.import_module(_PROVIDER_MODULES[name])
