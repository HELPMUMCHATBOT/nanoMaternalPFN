from .maternal_scm import sample_task

def sample_generic_task(**kwargs):
    return sample_task(prior='generic', **kwargs)
