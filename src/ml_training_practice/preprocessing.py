from sklearn.model_selection import train_test_split

# to jest po to aby rozdzielić dane na train i test, a nie trenować model na całym zbiorze danych, co mogłoby prowadzić do overfittingu i błędnej oceny modelu.
# test_size=0.2 oznacza, że 20% danych zostanie przeznaczone na testowanie modelu, a 80% na jego trenowanie.
# random_state=42 jest po to aby zapewnić powtarzalność wyników - dzięki temu za każdym razem, gdy uruchomimy ten kod, podział danych będzie taki sam.
def split_regression_data(features, target, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
