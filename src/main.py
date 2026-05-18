from train import main as trainTransformer
from model2 import main as trainBiGRU
from visualize import main as vm
from predict import main as pm


def _bad(choice):
    print(f"Invalid choice: {choice!r}")


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
            else:
                _bad(values)
        elif which == "2":
            print("Default values (1) or custom values (2)")
            values = input("choose: ").strip()
            print("Embeddings:")
            print("1: from scratch (default)")
            print("2: pretrained GloVe")
            emb_choice = input("choose: ").strip()
            if emb_choice not in {"1", "2"}:
                _bad(emb_choice)
                return
            use_pretrained = emb_choice == "2"

            glove_dim = 100
            freeze = False
            if use_pretrained:
                print("GloVe dim? (50/100/200/300, default 100)")
                gd = input("choose: ").strip()
                if gd in {"50", "100", "200", "300"}:
                    glove_dim = int(gd)
                print("Freeze embeddings? (y/N)")
                freeze = input("choose: ").strip().lower() == "y"

            if values == "1":
                if use_pretrained:
                    trainBiGRU(pretrained=True, glove_dim=glove_dim,
                               freeze_embeddings=freeze)
                else:
                    trainBiGRU()
            elif values == "2":
                kwargs = dict(
                    lr=float(input("lr (float): ")),
                    embed_dim=int(input("embed_dim (int): ")),
                    hidden_dim=int(input("hidden_dim (int): ")),
                    epochs=int(input("epochs (int): ")),
                    dropout=float(input("dropout (float): ")),
                )
                if use_pretrained:
                    kwargs.update(pretrained=True, glove_dim=glove_dim,
                                  freeze_embeddings=freeze)
                trainBiGRU(**kwargs)
            else:
                _bad(values)
        else:
            _bad(which)

    elif choice == "3":
        print("Which saved model?")
        print("1: model.pt (Transformer)")
        print("2: model2.pt (BiGRU, from scratch)")
        print("3: model2_glove_frozen.pt (BiGRU, frozen GloVe)")
        print("4: model2_glove_finetune.pt (BiGRU, fine-tuned GloVe)")
        which = input("choose: ").strip()
        paths = {
            "1": "models/model.pt",
            "2": "models/model2.pt",
            "3": "models/model2_glove_frozen.pt",
            "4": "models/model2_glove_finetune.pt",
        }
        if which in paths:
            pm(paths[which])
        else:
            _bad(which)
    else:
        _bad(choice)


if __name__ == "__main__":
    main()