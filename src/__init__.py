import git 


CURRENT_ROOT_DIR = git.Repo(__file__, search_parent_directories=True).working_tree_dir

