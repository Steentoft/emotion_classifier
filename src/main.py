from train import main as tm
from visualize import main as vm
from predict import main as pm

def main():
    print("What do you want to do?")
    print("\n")
    print("1: analyze data")
    print("2: train model")
    print("3: validate model")

    choice = input("choose: ")

    if choice == "1":
        vm()
    elif choice == "2":
        print("Default values (1) or custom values (2)")
        values = input("choose: ")
        if values == "1":
            tm()
        elif values == "2":
            tm(float(input("lr (float): ")), int(input("n_heads (int): ")), int(input("n_layers (int): ")))
    elif choice == "3":
        pm()


if __name__ == "__main__":
    main()
