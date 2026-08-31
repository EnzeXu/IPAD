from invariant_physics.dataset import get_dataset

if __name__ == "__main__":
    dataset = get_dataset()
    dataset.build()
    if dataset.args.extract_csv:
        dataset.extract_csv()
