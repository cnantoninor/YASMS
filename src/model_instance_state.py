import os


class ModelInstanceState:
    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory {self.directory} not found")
        if not os.path.isdir(self.directory):
            raise NotADirectoryError(f"{self.directory} is not a directory")
        parts = self.directory.split(os.path.sep)
        if len(parts) < 3:
            raise ValueError(
                "Directory path must have at least three parts: {modelType}/{modelName}/{modelInstanceDate}"
            )
        self.mod_type, self.mod_name, self.mod_instance_date = parts[-3:]

    @property
    def type(self):
        return self.mod_type

    @property
    def name(self):
        return self.mod_name

    @property
    def instance_date(self):
        return self.mod_instance_date
