def import_class_from_string(path: str):
    from importlib import import_module

    module_path, _, class_name = path.rpartition(".")
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass
