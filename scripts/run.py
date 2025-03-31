from loan_default_predictor.model import (
    SimpleNet,
    DeepNet,
    DeepNetWithSkipBN,
    DeepNetWithDropout,
)
from loan_default_predictor.data import load_data
from loan_default_predictor.train import train_model


def main():
    experiment = input("Введите номер эксперимента (1-5) или любую другую строку, чтобы выполнить все по очереди: ")

    X_train, y_train, X_test, y_test = load_data("loan_train.csv", "loan_test.csv")

    input_size = X_train.shape[1]

    # Значения по умолчанию
    config = {
        "epochs": 30,
        "batch_size": 32,
        "lr": 0.01,
        "seed": 42,
        "weight_decay": 0.0,
        "dropout_p": 0.0,
    }

    if experiment == "1":
        model = SimpleNet(input_size=input_size, hidden_size=32)
        config["epochs"] = 65 # опытным путём решили, что это оптимально

    elif experiment == "2":
        model = DeepNet(input_size=input_size, hidden_size=128)
        config["epochs"] = 40 # опытным путём решили, что это оптимально

    elif experiment == "3":
        model = DeepNetWithSkipBN(input_size=input_size, hidden_size=128)
        config["epochs"] = 45

    elif experiment == "4":
        dropout_values = [0.01, 0.1, 0.2, 0.5, 0.9]
        results = []

        for p in dropout_values:
            print(f"\n🚀 Запуск с dropout_p = {p}")
            model = DeepNetWithDropout(input_size=input_size, hidden_size=128, dropout_p=p)

            auc = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                epochs=45,
                batch_size=32,
                lr=0.01,
                seed=42,
                weight_decay=0.0,
                log_dir=f"runs/experiment_4/dropout_{p}",
                experiment=f"4_dropout{p}"
            )

            results.append((p, auc))

        # Найдём лучшую dropout по AUC
        best = max(results, key=lambda x: x[1])
        print(f"\n✅ ЛУЧШИЙ DROPOUT: p = {best[0]} → Test AUC = {best[1]:.4f}")

    elif experiment == "5":
        weight_decays = [0.1, 0.01, 0.001]
        learning_rates = [0.01, 0.05, 0.1]
        results = []

        for wd in weight_decays:
            for lr in learning_rates:
                print(f"\n🚀 Запуск с weight_decay={wd}, lr={lr}")
                model = DeepNetWithDropout(input_size=input_size, hidden_size=128, dropout_p=0.5) # dropout_p=0.5 показал себя лучше всего
                auc = train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=30,
                    batch_size=32,
                    lr=lr,
                    seed=42,
                    weight_decay=wd,
                    log_dir=f"runs/experiment_5/wd_{wd}_lr_{lr}",
                    experiment=f"5_wd{wd}_lr{lr}"
                )
                results.append((wd, lr, auc))

        # Найдём лучшую комбинацию по AUC
        best = max(results, key=lambda x: x[2])
        print(f"\n✅ ЛУЧШИЙ РЕЗУЛЬТАТ: weight_decay={best[0]}, lr={best[1]} → Test AUC = {best[2]:.4f}")

    else:
        print("⚠️ Пока доступны эксперименты 1–5. Выберите один из них.")
        return

    if experiment != '5' and experiment != '4':
        # Обучение
        train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            seed=config["seed"],
            weight_decay=config["weight_decay"],
            log_dir=f"runs/experiment_{experiment}",
            experiment=experiment
        )


if __name__ == "__main__":
    main()
