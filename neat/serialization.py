import pickle as pkl


def save_genome(genome, full_path):
    with open(full_path, 'wb') as f:
        pkl.dump(genome, f)
