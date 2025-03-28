import os

def check_folder(example):
    """
    Check for and create (if necessary) a subfolder for storing images based on the provided example identifier.

    This function constructs a path in the following structure:
        <current_working_directory>/images/images_<example>
    If the folder does not exist, it is created.

    Parameters:
    -----------
    example : str
        A string identifier for the example, used to create a unique subfolder name.

    Returns:
    --------
    str
        The full path to the image subfolder.
    """
    # Get the current working directory.
    cwd = os.getcwd()
    # Construct the target directory path.
    newpath = os.path.join(os.path.join(cwd, "images"), "images_" + example)
    
    # Check if the target directory exists; if not, create it.
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath
