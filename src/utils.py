import os


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self):
        # Will be important later for TF logging
        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())

        self.datasets = os.path.join(ProjectPath.base, "data", "datasets")

    def get_dataset(self, language, dataset):
        return os.path.join(self.datasets, language, language+"."+dataset)

project_path = ProjectPath()
