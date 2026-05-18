from train import main as trainTransformer
from model2 import main as trainBiGRU
from visualize import main as vm
from predict import main as pm


def main():
    print("What do you want to do?")
    print("1: analyze data")
    print("2: train model")
    print("3: validate model")

    choice = input("choose: ").strip()

    if choice == "1":
        vm()

    elif choice == "2":
        print("Which model?")
        print("1: Transformer (model1 -> model.pt)")
        print("2: BiGRU (model2 -> model2.pt)")
        which = input("choose: ").strip()

        if which == "1":
            print("Default values (1) or custom values (2)")
            values = input("choose: ").strip()
            if values == "1":
                trainTransformer()
            elif values == "2":
                trainTransformer(
                    float(input("lr (float): ")),
                    int(input("n_heads (int): ")),
                    int(input("n_layers (int): ")),
                )
        elif which == "2":
            print("Default values (1) or custom values (2)")
            values = input("choose: ").strip()
            if values == "1":
                trainBiGRU()
            elif values == "2":
                trainBiGRU(
                    lr=float(input("lr (float): ")),
                    embed_dim=int(input("embed_dim (int): ")),
                    hidden_dim=int(input("hidden_dim (int): ")),
                    epochs=int(input("epochs (int): ")),
                    dropout=float(input("dropout (float): ")),
                )

    elif choice == "3":
        print("Which saved model?")
        print("1: model.pt (Transformer)")
        print("2: model2.pt (BiGRU)")
        which = input("choose: ").strip()
        if which == "1":
            pm("model.pt")
        elif which == "2":
            pm("model2.pt")


if __name__ == "__main__":
    main()