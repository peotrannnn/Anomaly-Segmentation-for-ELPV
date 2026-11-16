from elpv_dataset.utils import load_dataset


def main():
    images, probabilities, cell_types = load_dataset()

    print("ELPV dataset loaded successfully.")
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Cell types shape: {cell_types.shape}")


if __name__ == "__main__":
    main()