def progress_bar(i, total):
    """
    Print a progress bar to the console.

    :param i: Current iteration index.
    :param total: Total number of iterations (optional).
    """

    percent = (i + 1) / total * 100
    bar_length = 80
    filled_length = int(bar_length * percent // 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {percent:.2f}%', end='\r')

    if i == total - 1:
        print(f'\rProgress: |{bar}| {percent:.2f}% - complete', end='\r')
        print()  # New line at the end of the progress bar